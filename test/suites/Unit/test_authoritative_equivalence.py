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

"""Authoritative-path pure-function equivalence (阶段 B, 4.4 C1 切新等价).

``_lookup_external_hit_tokens_authoritative`` 在 <NEW> 模式下用 ``resolve_hit``
决定链式 l,再经 ``chain_absolute_l`` 注入旧 Stage-2;本测试校验该纯函数链
(legacy_chain_candidate_l ↔ resolve_hit)在相同 lookup 输入下数字一致。
"""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
KVS_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "kv_spec_table.py"
BUILDER_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "spec_table_builder.py"


def _load_pure_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kv_spec_table = _load_pure_module("kv_spec_table", KVS_PATH)

# spec_table_builder imports vllm classes lazily inside functions only; but it
# does `from vllm.v1.kv_cache_interface import ...` at module top. Provide the
# same fake vllm namespace as test_spec_table_builder.
import types
from types import SimpleNamespace

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

ucm = types.ModuleType("ucm")
ucm.__path__ = []
integ = types.ModuleType("ucm.integration")
integ.__path__ = []
vllm_sub = types.ModuleType("ucm.integration.vllm")
vllm_sub.__path__ = []
sys.modules["ucm"] = ucm
sys.modules["ucm.integration"] = integ
sys.modules["ucm.integration.vllm"] = vllm_sub
sys.modules["ucm.integration.vllm.kv_spec_table"] = kv_spec_table

_builder_spec = importlib.util.spec_from_file_location(
    "ucm.integration.vllm.spec_table_builder", BUILDER_PATH
)
spec_table_builder = importlib.util.module_from_spec(_builder_spec)
sys.modules[_builder_spec.name] = spec_table_builder
_builder_spec.loader.exec_module(spec_table_builder)

CacheKind = kv_spec_table.CacheKind
SpecRow = kv_spec_table.SpecRow
SpecTable = kv_spec_table.SpecTable
resolve_hit = kv_spec_table.resolve_hit
legacy_chain_candidate_l = spec_table_builder.legacy_chain_candidate_l


def _lookup_present(block_ids, present):
    last = -1
    for i, bid in enumerate(block_ids):
        if bid not in present:
            break
        last = i
    return last


class AuthoritativeEquivalenceTest(unittest.TestCase):
    """阶段 B: <NEW> 链式 l == legacy Stage-1 候选(相同 lookup 输入)。"""

    def _qwen3_8_shape(self):
        # Qwen3.8-27B GDN 形态: 单 full-attn 组(block 1536)+ mamba 状态组。
        groups = [
            _group(fake.FullAttentionSpec(1536), ("m0",)),
            _group(fake.MambaSpec(64, "align"), ("m1",)),
        ]
        spec = spec_table_builder.build_spec_table(groups)
        self.assertEqual(spec.row("fa0").block_size, 1536)
        self.assertEqual(spec.row("fa0").kind, CacheKind.CHAIN)
        self.assertEqual(spec.row("mamba1").kind, CacheKind.SNAPSHOT)
        return spec

    def test_chain_l_resolve_equals_legacy(self):
        spec = self._qwen3_8_shape()
        row_to_gid = {"fa0": 0, "mamba1": 1}
        group_block_ids = [[bytes([i]) for i in range(8)], [bytes([i]) for i in range(4)]]
        present = {bytes([0]), bytes([1])}  # 前两块存在

        def existence_by_chain(row, block_ids):
            gi = row_to_gid[row.group_name]
            bs = row.block_size
            external = group_block_ids[gi][0 // bs:]
            if not external:
                return 0
            hit_blocks = _lookup_present(external, present) + 1
            return 0 + max(hit_blocks, 0) * bs

        new_l, _ = resolve_hit(spec, {}, existence_by_chain, {})
        legacy_l = legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lambda ids: _lookup_present(ids, present),
            block_sizes=[1536],
            lcm_block_size=1536,
        )
        # 两块 1536 存在 +1 -> 2*1536=3072。
        self.assertEqual(new_l, 3072)
        self.assertEqual(new_l, legacy_l)

    def test_align_floor_when_group_sizes_differ(self):
        spec = SpecTable(
            [
                SpecRow("fa0", CacheKind.CHAIN, 128),
                SpecRow("fa1", CacheKind.CHAIN, 64),
            ]
        )
        row_to_gid = {"fa0": 0, "fa1": 1}
        group_block_ids = [[bytes([i]) for i in range(16)]] * 2
        present_g0 = {bytes([0]), bytes([1]), bytes([2])}
        present_g1 = {bytes([0]), bytes([1])}

        def existence_by_chain(row, block_ids):
            gi = row_to_gid[row.group_name]
            bs = row.block_size
            external = group_block_ids[gi][0 // bs:]
            if not external:
                return 0
            present = present_g0 if gi == 0 else present_g1
            hit_blocks = _lookup_present(external, present) + 1
            return 0 + max(hit_blocks, 0) * bs

        new_l, _ = resolve_hit(spec, {}, existence_by_chain, {})
        legacy_l = legacy_chain_candidate_l(
            num_computed_tokens=0,
            full_attn_group_ids=[0, 1],
            group_block_ids=group_block_ids,
            lookup_on_prefix=lambda ids: _lookup_present(ids, present_g0),
            block_sizes=[128, 64],
            lcm_block_size=128,
        )
        # g0: 3 块 -> 384;g1: 2 块 -> 128;min=128,floor LCM=128。
        self.assertEqual(new_l, legacy_l)
        self.assertEqual(new_l, 128)

    def test_miss_equals_zero(self):
        spec = self._qwen3_8_shape()
        row_to_gid = {"fa0": 0, "mamba1": 1}
        group_block_ids = [[bytes([i]) for i in range(8)], [bytes([i]) for i in range(4)]]

        def existence_by_chain(row, block_ids):
            gi = row_to_gid[row.group_name]
            bs = row.block_size
            external = group_block_ids[gi][0 // bs:]
            if not external:
                return 0
            hit_blocks = _lookup_present(external, set()) + 1
            return 0 + max(hit_blocks, 0) * bs

        new_l, _ = resolve_hit(spec, {}, existence_by_chain, {})
        self.assertEqual(new_l, 0)


def _group(spec, layer_names=("layer0",)):
    return SimpleNamespace(layer_names=layer_names, kv_cache_spec=spec)


if __name__ == "__main__":
    unittest.main()