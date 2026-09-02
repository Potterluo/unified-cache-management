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
        def __init__(self, block_size):
            self.block_size = block_size

    class MLAAttentionSpec(KVCacheSpec):
        def __init__(self, block_size):
            self.block_size = block_size

    class MambaSpec(KVCacheSpec):
        def __init__(self, block_size, mamba_cache_mode="align"):
            self.block_size = block_size
            self.mamba_cache_mode = mamba_cache_mode

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


class DoubleRunFlagTest(unittest.TestCase):
    def test_flag_parsing(self):
        self.assertFalse(spec_table_builder.spec_table_double_run_enabled())
        os.environ["UCM_SPEC_TABLE_DOUBLE_RUN"] = "1"
        try:
            self.assertTrue(spec_table_builder.spec_table_double_run_enabled())
        finally:
            del os.environ["UCM_SPEC_TABLE_DOUBLE_RUN"]


if __name__ == "__main__":
    unittest.main()
