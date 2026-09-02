#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Unit tests for the spec-table double-run builder (4.4 C1 双跑记账).

``spec_table_builder`` is only imported by the vLLM connectors at runtime, so it
is loaded here with a stubbed ``vllm.v1.kv_cache_interface`` (fake spec classes)
and a stubbed ``ucm`` package namespace pointing at the real, zero-dependency
``kv_spec_table`` module -- without importing the built C++ extension.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
KVS_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "kv_spec_table.py"
BUILDER_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "spec_table_builder.py"


def _load_pure_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_vllm():
    """Fake the vllm KV-cache interface classes used by the builder."""
    fake = SimpleNamespace()

    class KVCacheSpec:
        block_size = None

    class FullAttentionSpec(KVCacheSpec):
        def __init__(self, block_size, sliding_window=None, compress_ratio=None):
            self.block_size = block_size
            if sliding_window is not None:
                self.sliding_window = sliding_window
            if compress_ratio is not None:
                self.compress_ratio = compress_ratio

    class MLAAttentionSpec(KVCacheSpec):
        def __init__(self, block_size, sliding_window=None):
            self.block_size = block_size
            if sliding_window is not None:
                self.sliding_window = sliding_window

    class MambaSpec(KVCacheSpec):
        def __init__(self, block_size, mamba_cache_mode="align", sliding_window=None):
            self.block_size = block_size
            self.mamba_cache_mode = mamba_cache_mode
            if sliding_window is not None:
                self.sliding_window = sliding_window

    class UniformTypeKVCacheSpecs:
        def __init__(self, kv_cache_specs):
            self.kv_cache_specs = kv_cache_specs

    fake.KVCacheSpec = KVCacheSpec
    fake.FullAttentionSpec = FullAttentionSpec
    fake.MLAAttentionSpec = MLAAttentionSpec
    fake.MambaSpec = MambaSpec
    fake.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs

    pkg = types.ModuleType("vllm")
    pkg.__path__ = []
    sub = types.ModuleType("vllm.v1")
    sub.__path__ = []
    leaf = types.ModuleType("vllm.v1.kv_cache_interface")
    leaf.KVCacheSpec = KVCacheSpec
    leaf.FullAttentionSpec = FullAttentionSpec
    leaf.MLAAttentionSpec = MLAAttentionSpec
    leaf.MambaSpec = MambaSpec
    leaf.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
    sys.modules["vllm"] = pkg
    sys.modules["vllm.v1"] = sub
    sys.modules["vllm.v1.kv_cache_interface"] = leaf
    return fake


def _install_ucm_namespace(kv_spec_module):
    ucm = types.ModuleType("ucm")
    ucm.__path__ = []
    integ = types.ModuleType("ucm.integration")
    integ.__path__ = []
    vllm_sub = types.ModuleType("ucm.integration.vllm")
    vllm_sub.__path__ = []
    sys.modules["ucm"] = ucm
    sys.modules["ucm.integration"] = integ
    sys.modules["ucm.integration.vllm"] = vllm_sub
    sys.modules["ucm.integration.vllm.kv_spec_table"] = kv_spec_module


# Load once at import time (module-level, same pattern as test_kv_spec_table).
kv_spec_table = _load_pure_module("kv_spec_table", KVS_PATH)
fake_vllm = _install_fake_vllm()
_install_ucm_namespace(kv_spec_table)
_builder_spec = importlib.util.spec_from_file_location(
    "ucm.integration.vllm.spec_table_builder", BUILDER_PATH
)
spec_table_builder = importlib.util.module_from_spec(_builder_spec)
sys.modules[_builder_spec.name] = spec_table_builder
_builder_spec.loader.exec_module(spec_table_builder)

CacheKind = kv_spec_table.CacheKind
RankRule = kv_spec_table.RankRule


def _group(spec, layer_names=("layer0",)):
    return SimpleNamespace(layer_names=layer_names, kv_cache_spec=spec)


class BuildSpecTableTest(unittest.TestCase):
    def test_kind_and_block_size_mapping(self):
        # mamba(快照)/ MLA(链式,all_union)/ full-attn(链式,all_intersect)。
        groups = [
            _group(fake_vllm.MLAAttentionSpec(128), ("m0",)),
            _group(fake_vllm.MambaSpec(64, "align"), ("m1",)),
            _group(fake_vllm.FullAttentionSpec(128), ("m2",)),
        ]
        spec = spec_table_builder.build_spec_table(groups)
        self.assertEqual([r.group_name for r in spec.rows], ["mla0", "mamba1", "fa2"])
        self.assertEqual(
            [r.kind for r in spec.rows],
            [
                CacheKind.CHAIN,
                CacheKind.SNAPSHOT,
                CacheKind.CHAIN,
            ],
        )
        self.assertEqual(spec.row("mla0").rank_rule, RankRule.ALL_UNION)
        self.assertEqual(spec.row("fa2").rank_rule, RankRule.ALL_INTERSECT)
        self.assertEqual(spec.row("mamba1").retention.grid_alignment, 64)
        # LCM 只统计链式组(mamba 快照组的 64 不参与对齐,4.2)。
        self.assertEqual(spec.lcm_block_size, 128)

    def test_ascend_c4_compress_block_size_token_scale(self):
        # DSV4 C4 压缩组: 引擎 spec.block_size=32(storage 刻度) + compress_ratio=4
        # -> 规格表 block 应为 32*4=128(逻辑 token 刻度,与 FAWA 旧表
        # token_block_size 对齐,4.1/6.2/4.4 C1)。
        class AscendC4Spec(fake_vllm.FullAttentionSpec):
            def __init__(self, block_size, compress_ratio=1):
                super().__init__(block_size)
                self.compress_ratio = compress_ratio

        groups = [_group(AscendC4Spec(32, compress_ratio=4), ("m0",))]
        spec = spec_table_builder.build_spec_table(groups)
        self.assertEqual(spec.rows[0].kind, CacheKind.CHAIN)
        self.assertEqual(spec.rows[0].block_size, 128)
        self.assertEqual(spec.lcm_block_size, 128)
        # FAWA 旧表 KVCacheGroupMeta.token_block_size = 128 -> 双跑无 block 告警。
        legacy = [SimpleNamespace(group_id=0, token_block_size=128)]
        self.assertEqual(spec_table_builder.double_run_ledger(spec, legacy), [])

    def test_non_compress_group_unchanged(self):
        # 无 compress_ratio 的组: block 保持引擎原始值(不缩放)。
        groups = [_group(fake_vllm.FullAttentionSpec(128), ("m0",))]
        spec = spec_table_builder.build_spec_table(groups)
        self.assertEqual(spec.rows[0].block_size, 128)

    def test_uniform_type_spec_recursion(self):
        # 混合 UniformType 组(MLA + mamba): 只要含 mamba(align) 层即为快照组。
        uni = fake_vllm.UniformTypeKVCacheSpecs(
            {
                "a": fake_vllm.MLAAttentionSpec(128),
                "b": fake_vllm.MambaSpec(64, "align"),
            }
        )
        spec = spec_table_builder.build_spec_table([_group(uni)])
        self.assertEqual(spec.rows[0].kind, CacheKind.SNAPSHOT)
        # 阶段 1 记账口径: block_size 与旧 GroupInfo 同源(取首个成员);
        # 快照网格取成员级 block_size 属阶段 2(SnapshotStore)精度。
        self.assertEqual(spec.rows[0].block_size, 128)

    def test_pure_mamba_group_block_size(self):
        spec = spec_table_builder.build_spec_table(
            [_group(fake_vllm.MambaSpec(64, "align"))]
        )
        self.assertEqual(spec.rows[0].kind, CacheKind.SNAPSHOT)
        self.assertEqual(spec.rows[0].block_size, 64)
        self.assertEqual(spec.rows[0].retention.grid_alignment, 64)

    def test_group_seeds_recorded(self):
        groups = [_group(fake_vllm.FullAttentionSpec(128))]
        spec = spec_table_builder.build_spec_table(groups, group_seeds=["deadbeef"])
        self.assertEqual(spec.row("fa0").seed, "deadbeef")


class DoubleRunLedgerTest(unittest.TestCase):
    def _legacy_group_info(self, group_id, block_size, is_mamba_align=False):
        # 镜像 hla_connector.GroupInfo 的字段(记账只读 getattr)。
        return SimpleNamespace(
            group_id=group_id,
            block_size=block_size,
            is_mamba_align=is_mamba_align,
            layer_names=(f"layer{group_id}",),
            seed=b"seed",
        )

    def test_no_mismatch_when_mapping_aligned(self):
        groups = [
            _group(fake_vllm.MLAAttentionSpec(128)),
            _group(fake_vllm.MambaSpec(64, "align")),
        ]
        spec = spec_table_builder.build_spec_table(groups)
        legacy = [
            self._legacy_group_info(0, 128),
            self._legacy_group_info(1, 64, is_mamba_align=True),
        ]
        self.assertEqual(spec_table_builder.double_run_ledger(spec, legacy), [])

    def test_mismatch_on_block_size_and_kind(self):
        groups = [
            _group(fake_vllm.MLAAttentionSpec(128)),
            _group(fake_vllm.MambaSpec(64, "align")),
        ]
        spec = spec_table_builder.build_spec_table(groups)
        # block_size 不一致(group0: 96 vs 128)+ kind 不一致(group1 旧逻辑认为非 mamba)。
        legacy = [
            self._legacy_group_info(0, 96),
            self._legacy_group_info(1, 64, is_mamba_align=False),
        ]
        msgs = spec_table_builder.double_run_ledger(spec, legacy)
        self.assertEqual(len(msgs), 2)
        self.assertTrue(any("block_size" in m for m in msgs))
        self.assertTrue(any("kind" in m for m in msgs))

    def test_mismatch_on_group_count(self):
        groups = [_group(fake_vllm.MLAAttentionSpec(128))]
        spec = spec_table_builder.build_spec_table(groups)
        extra = [self._legacy_group_info(0, 128), self._legacy_group_info(1, 128)]
        msgs = spec_table_builder.double_run_ledger(spec, extra)
        self.assertTrue(any("组数量不一致" in m for m in msgs))

    def test_fawa_kv_cache_group_meta_aligned(self):
        # hma_connector.KVCacheGroupMeta: 字段是 token_block_size,无 is_mamba_align。
        groups = [
            _group(fake_vllm.FullAttentionSpec(256), ("m0",)),
            _group(fake_vllm.FullAttentionSpec(64), ("m1",)),
        ]
        spec = spec_table_builder.build_spec_table(groups)
        legacy = [
            SimpleNamespace(group_id=0, token_block_size=256),
            SimpleNamespace(group_id=1, token_block_size=64),
        ]
        # 不改写旧表语义: 块大小比对通过,kind 比对在缺少 is_mamba_align 时跳过。
        self.assertEqual(spec_table_builder.double_run_ledger(spec, legacy), [])

    def test_fawa_kv_cache_group_meta_block_size_mismatch(self):
        groups = [_group(fake_vllm.FullAttentionSpec(256))]
        spec = spec_table_builder.build_spec_table(groups)
        legacy = [SimpleNamespace(group_id=0, token_block_size=128)]
        msgs = spec_table_builder.double_run_ledger(spec, legacy)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(any("block_size" in m for m in msgs))


class DoubleRunFlagTest(unittest.TestCase):
    def test_flag_parsing(self):
        self.assertFalse(spec_table_builder.spec_table_double_run_enabled())
        os.environ["UCM_SPEC_TABLE_DOUBLE_RUN"] = "1"
        try:
            self.assertTrue(spec_table_builder.spec_table_double_run_enabled())
        finally:
            del os.environ["UCM_SPEC_TABLE_DOUBLE_RUN"]

    def test_authoritative_flag_parsing(self):
        self.assertFalse(spec_table_builder.spec_table_authoritative_enabled())
        os.environ["UCM_SPEC_TABLE_AUTHORITATIVE"] = "true"
        try:
            self.assertTrue(spec_table_builder.spec_table_authoritative_enabled())
        finally:
            del os.environ["UCM_SPEC_TABLE_AUTHORITATIVE"]
        os.environ["UCM_SPEC_TABLE_AUTHORITATIVE"] = "off"
        self.assertFalse(spec_table_builder.spec_table_authoritative_enabled())
        del os.environ["UCM_SPEC_TABLE_AUTHORITATIVE"]


class LegacyChainCandidateTest(unittest.TestCase):
    """4.4 C1 记账基准: legacy_chain_candidate_l 纯函数与旧 Stage-1 数学一致。

    回归点: hla_connector._double_run_shadow_resolve 曾把 legacy 链式候选的
    初值设成 num_computed_tokens,导致 min 恒等于初值、记账永远为 0
    (真实模型日志: resolve_hit l=3072 != legacy chain l=0)。抽成纯函数后,
    此处用与 4.5 算例 A 同构的数字直接断言。
    """

    @staticmethod
    def _lookup_present(block_ids, present: set[int]) -> int:
        """模拟 store.lookup_on_prefix: 返回最后一个连续存在块的下标,无则 -1。"""
        last = -1
        for i, bid in enumerate(block_ids):
            if bid not in present:
                break
            last = i
        return last

    def _block_ids(self, n: int) -> list[bytes]:
        return [bytes([i]) for i in range(n)]

    def test_example_A_numbers(self):
        # 与 4.5 算例 A 同构: mla(128x36 块存在) / csa(128x35 块存在) / swa(全在)。
        group_block_ids = [
            self._block_ids(64),  # group 0 = mla
            self._block_ids(64),  # group 1 = csa
            self._block_ids(64),  # group 2 = swa
        ]
        present_mla = {bytes([i]) for i in range(36)}
        present_csa = {bytes([i]) for i in range(35)}
        present_swa = {bytes([i]) for i in range(64)}

        def make_lookup(present):
            return lambda ids: LegacyChainCandidateTest._lookup_present(ids, present)

        l = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0, 1, 2],
            group_block_ids=group_block_ids,
            lookup_on_prefix=make_lookup(present_mla),
            block_sizes=[128, 128, 128],
            lcm_block_size=128,
        )
        # 只用一个组: mla 36 块 -> 4608,floor 128 -> 4608。
        self.assertEqual(l, 4608)

        # 三组 min: mla=4608, csa=4480, swa=8192 -> min=4480,对齐 128。
        l = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0, 1, 2],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lambda ids: min(
                make_lookup(present_mla)(ids),
                make_lookup(present_csa)(ids),
                make_lookup(present_swa)(ids),
            ),
            block_sizes=[128, 128, 128],
            lcm_block_size=128,
        )
        self.assertEqual(l, 4480)

    def test_regression_3072_not_stuck_at_zero(self):
        # Qwen3.8 形态: 单 full-attn 组 block=1536,两块都在 -> 2*1536=3072;
        # 旧记账曾因 min 初值 bug 恒为 0。抽成纯函数后必须返回 3072。
        group_block_ids = [self._block_ids(8)]
        lookup = lambda ids: self._lookup_present(ids, {bytes([0]), bytes([1])})
        l = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lookup,
            block_sizes=[1536],
            lcm_block_size=1536,
        )
        self.assertEqual(l, 3072)

    def test_miss_returns_zero(self):
        group_block_ids = [self._block_ids(8)]
        lookup = lambda ids: self._lookup_present(ids, set())  # 全 miss
        l = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lookup,
            block_sizes=[1536],
            lcm_block_size=1536,
        )
        self.assertEqual(l, 0)

    def test_num_computed_offset(self):
        # num_computed_tokens 不为 0 时,候选长度是绝对位置(旧逻辑的
        # external_hit_tokens 相对值 + num_computed)。
        # num_computed=1536 -> fa_hbm_blocks=1,external 从第 2 块开始;
        # store 里第 2 块存在 -> hit_blocks=1 -> 相对 1536 -> 绝对 3072。
        group_block_ids = [self._block_ids(16)]
        l = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=1536,
            full_attn_group_ids=[0],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lambda ids: self._lookup_present(ids, {bytes([1])}),
            block_sizes=[1536],
            lcm_block_size=1536,
        )
        self.assertEqual(l, 3072)

    def test_min_over_groups_and_floor_lcm(self):
        # 多组不同 block_size: min 后向下对齐 LCM(4.2)。
        group_block_ids = [
            self._block_ids(16),
            self._block_ids(16),
        ]
        l = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0, 1],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lambda ids: self._lookup_present(
                ids, {bytes([0]), bytes([1]), bytes([2])}
            ),
            block_sizes=[128, 64],
            lcm_block_size=128,
        )
        # g0 3 块(128) -> 384;g1 3 块(64) -> 192;min=192,floor LCM=128。
        self.assertEqual(l, 128)


class ResolveVsLegacyEquivalenceTest(unittest.TestCase):
    """4.4 C1 核心: resolve_hit(新) 与 legacy 链式候选(旧) 数字等价。

    Shadow resolve 的 existence_by_chain 必须与 legacy_chain_candidate_l 对
    同一份 lookup 结果给出同一长度;不等价即双跑记账失真,冻结切新(4.4 C1)。
    """

    def _chain_spec(self):
        return kv_spec_table.SpecTable(
            [
                kv_spec_table.SpecRow("fa0", CacheKind.CHAIN, 128),
                kv_spec_table.SpecRow("fa1", CacheKind.CHAIN, 64),
            ]
        )

    def test_equivalent_on_shared_lookup(self):
        spec = self._chain_spec()
        row_to_gid = {"fa0": 0, "fa1": 1}
        group_block_ids = [[bytes([i]) for i in range(16)]] * 2

        def present_gid0(ids):
            last = -1
            for i, bid in enumerate(ids):
                if i >= 3:
                    break
                last = i
            return last

        def present_gid1(ids):
            last = -1
            for i, bid in enumerate(ids):
                if i >= 2:
                    break
                last = i
            return last

        lookup = {0: present_gid0, 1: present_gid1}

        def existence_by_chain(row, block_ids):
            gi = row_to_gid[row.group_name]
            bs = row.block_size
            external = group_block_ids[gi][0 // bs :]
            if not external:
                return 0
            hit_blocks = lookup[gi](external) + 1
            return 0 + max(hit_blocks, 0) * bs

        l_new, _ = kv_spec_table.resolve_hit(
            spec, {}, existence_by_chain, {}
        )
        l_legacy = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0, 1],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lambda ids: lookup[0](ids),
            block_sizes=[128, 64],
            lcm_block_size=128,
        )
        # g0 命中 3 块 -> 384;g1 命中 2 块 -> 128;min=128,floor LCM=128。
        self.assertEqual(l_new, l_legacy)
        self.assertEqual(l_new, 128)

    def test_equivalent_miss_side(self):
        spec = self._chain_spec()
        row_to_gid = {"fa0": 0, "fa1": 1}
        group_block_ids = [[bytes([i]) for i in range(16)]] * 2
        lookup = {
            0: lambda ids: -1,
            1: lambda ids: -1,
        }

        def existence_by_chain(row, block_ids):
            gi = row_to_gid[row.group_name]
            bs = row.block_size
            external = group_block_ids[gi][0 // bs :]
            if not external:
                return 0
            hit_blocks = lookup[gi](external) + 1
            return 0 + max(hit_blocks, 0) * bs

        l_new, _ = kv_spec_table.resolve_hit(spec, {}, existence_by_chain, {})
        l_legacy = spec_table_builder.legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0, 1],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lambda ids: -1,
            block_sizes=[128, 64],
            lcm_block_size=128,
        )
        self.assertEqual(l_new, l_legacy)
        self.assertEqual(l_new, 0)


class WindowGroupSpecTest(unittest.TestCase):
    """4.1 归属轴: sliding_window/tail_tokens 由规格表行数据承载(去硬编码)。"""

    def _fa_spec(self, block_size=128):
        return fake_vllm.FullAttentionSpec(block_size=block_size)

    def _swa_spec(self, block_size=32, window=2048):
        return fake_vllm.MLAAttentionSpec(block_size=block_size, sliding_window=window)

    def _ascend_c4_spec(self, block_size=32, compress_ratio=4):
        # DS-V4 C4 FA 组: 无窗口 + 压缩比 4(引擎 storage 刻度 32)。
        return fake_vllm.FullAttentionSpec(
            block_size=block_size, compress_ratio=compress_ratio
        )

    def test_fa_group_has_no_window(self):
        spec = self._fa_spec()
        self.assertIsNone(
            spec_table_builder.sliding_window_from_spec(spec),
            "FA 组归属轴必须为 None",
        )
        self.assertIsNone(
            spec_table_builder.tail_tokens_from_spec(spec, ("layers.0.kv_cache",), None)
        )

    def test_swa_group_window_and_tail(self):
        spec = self._swa_spec(window=2048)
        self.assertEqual(spec_table_builder.sliding_window_from_spec(spec), 2048)
        # swa_cache 张量保留完整窗口尾。
        self.assertEqual(
            spec_table_builder.tail_tokens_from_spec(
                spec, ("layers.2.swa_cache",), (32, 64)
            ),
            2048,
        )

    def test_compressor_group_tail_discounts_layer_ratio(self):
        spec = self._swa_spec(window=4096)
        # compressor 状态组只保留未压缩尾: 窗口 - 层压缩比(layer 2 = 32)。
        self.assertEqual(
            spec_table_builder.tail_tokens_from_spec(
                spec, ("layers.2.compressor_kv_cache",), (8, 16, 32)
            ),
            4096 - 32,
        )

    def test_uniform_member_window_must_agree(self):
        mixed = fake_vllm.UniformTypeKVCacheSpecs(
            {"a": self._swa_spec(window=2048), "b": self._swa_spec(window=1024)}
        )
        with self.assertRaises(ValueError):
            spec_table_builder.sliding_window_from_spec(mixed)
        agreed = fake_vllm.UniformTypeKVCacheSpecs(
            {"a": self._swa_spec(window=2048), "b": self._fa_spec()}
        )
        self.assertEqual(
            spec_table_builder.sliding_window_from_spec(agreed), 2048
        )

    def test_build_spec_table_carries_ownership_columns(self):
        groups = [
            _group(
                self._ascend_c4_spec(),
                layer_names=("layers.0.kv_cache",),
            ),
            _group(
                self._swa_spec(window=2048),
                layer_names=("layers.2.swa_cache",),
            ),
            _group(
                self._swa_spec(window=4096),
                layer_names=("layers.2.compressor_kv_cache",),
            ),
        ]
        table = spec_table_builder.build_spec_table(
            groups, layer_compress_ratios=(4, 4, 32)
        )
        fa, swa, comp = table.rows
        self.assertIsNone(fa.sliding_window)
        self.assertIsNone(fa.tail_tokens)
        self.assertEqual(swa.sliding_window, 2048)
        self.assertEqual(swa.tail_tokens, 2048)
        self.assertEqual(comp.sliding_window, 4096)
        self.assertEqual(comp.tail_tokens, 4096 - 32)

    def test_wa_row_without_tail_data_fails_loudly(self):
        # 无 compress_ratios 时 compressor 尾长无法折算 => 显式失败而非静默硬编码。
        groups = [
            _group(
                self._swa_spec(window=4096),
                layer_names=("layers.5.compressor_kv_cache",),
            )
        ]
        with self.assertRaises(AssertionError):
            spec_table_builder.build_spec_table(groups, layer_compress_ratios=None)


if __name__ == "__main__":
    unittest.main()
