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

"""FAWA 组归属去硬编码回归: 规格表驱动 vs 旧启发式等价。

提取 ``hma_connector.UCMFAWAConnector`` 的组归属方法(AST 免依赖模式,同
``test_kv_cache_layout``):

- ``can_handle_ascend_kv_cache_config`` / ``_get_ascend_base_block_size``:
  Ascend 布局能力探测;
- ``_init_group_metas``: FA/WA 归属与每组 canonical 尾长的消费逻辑。

断言两组关键性质:

1. 规格表驱动(Double-run/Authoritative 开)与旧启发式(Fallback)产出
   **完全一致**的 fa_group_ids / window_group_ids / group_metas —— 双跑记账
   之外的行为等价回归;
2. 归属轴由规格表行数据承载(sliding_window / tail_tokens),Connector 侧的
   tensor 名后缀 / 层压缩比启发式只在无规格表时兜底。
"""

import ast
import importlib.util
import re
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
HMA_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "hma_connector.py"
KVS_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "kv_spec_table.py"
BUILDER_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "spec_table_builder.py"


def _load_pure_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, *_args, **_kwargs):
        pass

    def info_once(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


logger = FakeLogger()


def extract_layer_index(name: str) -> int:
    match = re.search(r"layers\.(\d+)", name)
    assert match is not None, f"no layer index in {name!r}"
    return int(match.group(1))


def round_up(x: int, unit: int) -> int:
    return ((x + unit - 1) // unit) * unit


def _load_fawa_methods():
    """从 hma_connector 提取组归属相关符号,免去 vllm/torch 依赖。"""
    source = HMA_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    def is_assignment(node, names):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        )

    group_meta_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "KVCacheGroupMeta"
    )
    class_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "UCMFAWAConnector"
    )
    keep_methods = {
        "can_handle_ascend_kv_cache_config",
        "_get_ascend_base_block_size",
        "_init_group_metas",
    }
    body = [
        n
        for n in class_node.body
        if (
            isinstance(n, ast.FunctionDef) and n.name in keep_methods
        )
        or (
            is_assignment(
                n,
                {
                    "DEFAULT_HASH_BLOCK_SIZE",
                    "ASCEND_SUPPORTED_VLLM_BLOCK_SIZES",
                    "ASCEND_C4_COMPRESS_RATIO",
                },
            )
        )
    ]
    method_module = ast.Module(
        body=[group_meta_node, ast.ClassDef(name="UCMFAWAConnector", bases=[],
                                            keywords=[], body=body,
                                            decorator_list=[])],
        type_ignores=[],
    )
    ast.fix_missing_locations(method_module)
    namespace = {
        "List": list,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "dataclass": dataclass,
        "extract_layer_index": extract_layer_index,
        "round_up": round_up,
        "logger": logger,
        "KVCacheGroupMeta": None,  # defined above in the extracted module
    }
    compiled = compile(method_module, str(HMA_PATH), "exec")
    exec(compiled, namespace)
    return namespace["UCMFAWAConnector"]


UCMFAWAConnector = _load_fawa_methods()


class AscendSlidingWindowMLASpec:
    """DS-V4 Ascend 组规格最小 fake: 类名即引擎布局标签。

    类名必须与真实引擎一致(``can_handle_ascend_kv_cache_config`` 按
    ``type(spec).__name__`` 探测 Ascend 布局)。
    """

    def __init__(self, block_size, sliding_window=None, compress_ratio=None):
        self.block_size = block_size
        if sliding_window is not None:
            self.sliding_window = sliding_window
        if compress_ratio is not None:
            self.compress_ratio = compress_ratio


def _vllm_config(compress_ratios: Tuple[int, ...]):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=list(compress_ratios)),
        ),
        speculative_config=None,
    )


def _dsv4_like_groups():
    # DS-V4 Flash 简化布局(4 组): C4 FA / C128 FA / SWA 窗口 / compressor 窗口。
    groups = [
        # C4 FA: storage 32, 压缩比 4 -> logical 128(基准 = hash_block_size)。
        SimpleNamespace(
            layer_names=("layers.0.kv_cache",),
            kv_cache_spec=AscendSlidingWindowMLASpec(block_size=32, compress_ratio=4),
        ),
        # C128 FA: logical 4096。
        SimpleNamespace(
            layer_names=("layers.2.kv_cache",),
            kv_cache_spec=AscendSlidingWindowMLASpec(block_size=32, compress_ratio=128),
        ),
        # SWA 窗口: 保留完整窗口尾 2048。
        SimpleNamespace(
            layer_names=("layers.3.swa_cache",),
            kv_cache_spec=AscendSlidingWindowMLASpec(block_size=32, compress_ratio=4,
                                      sliding_window=2048),
        ),
        # compressor 窗口: 只保留未压缩尾 4096 - 层压缩比(32)。
        SimpleNamespace(
            layer_names=("layers.5.compressor_kv_cache",),
            kv_cache_spec=AscendSlidingWindowMLASpec(block_size=32, compress_ratio=4,
                                      sliding_window=4096),
        ),
    ]
    # layer 5(compressor 窗口组)的层压缩比定为 32,其余层 4。
    compress_ratios = tuple((4,) * 5 + (32,) + (4,) * (43 - 6))
    return groups, compress_ratios


def _connector(groups, compress_ratios, spec_table):
    conn = UCMFAWAConnector.__new__(UCMFAWAConnector)
    # 预置 __init__ 中 group metas 之前的初始状态(AST 提取不含 __init__)。
    conn.hash_block_size = UCMFAWAConnector.DEFAULT_HASH_BLOCK_SIZE
    conn.max_token_block_size = 0
    conn.group_metas = {}
    conn.is_ascend_layout = False
    conn.ascend_base_block_size = None
    conn.file_size = {}
    conn._kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    conn._vllm_config = _vllm_config(compress_ratios)
    conn._spec_table_double_run = spec_table
    return conn


def _install_fake_vllm():
    """Fake 最小 vllm 命名空间(same pattern as test_spec_table_builder)。

    独立加载 builder 所需: kv_cache_interface 四个 spec 类 +
    model_executor.models.utils.extract_layer_index。不 import 真 vllm,
    避免与同进程其他测试的 fake 注入互相干扰。
    """
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
_install_fake_vllm()
_install_ucm_namespace(kv_spec_table)
_builder_spec = importlib.util.spec_from_file_location(
    "ucm.integration.vllm.spec_table_builder", BUILDER_PATH
)
spec_table_builder = importlib.util.module_from_spec(_builder_spec)
sys.modules[_builder_spec.name] = spec_table_builder
_builder_spec.loader.exec_module(spec_table_builder)


class FawaSpecTableDrivenGroupsTest(unittest.TestCase):
    """规格表驱动归属与旧启发式在 DS-V4 布局上等价。"""

    def _run_both_modes(self):
        groups, compress_ratios = _dsv4_like_groups()

        # 规格表驱动(双跑/权威模式): 行序 = 组序。
        spec_driven = _connector(groups, compress_ratios, None)
        spec_driven._init_group_metas()

        # Fallback(开关全关): 旧启发式。
        spec_table = spec_table_builder.build_spec_table(
            groups, layer_compress_ratios=compress_ratios
        )
        legacy = _connector(groups, compress_ratios, spec_table)
        legacy._init_group_metas()

        return spec_driven, legacy

    def test_ascend_layout_detected(self):
        groups, compress_ratios = _dsv4_like_groups()
        conn = _connector(groups, compress_ratios, None)
        conn._init_group_metas()
        self.assertTrue(conn.is_ascend_layout)
        self.assertEqual(conn.ascend_base_block_size, 32)
        self.assertEqual(conn.hash_block_size, 128)

    def test_spec_driven_matches_legacy_ownership(self):
        spec_driven, legacy = self._run_both_modes()
        self.assertEqual(spec_driven.fa_group_ids, legacy.fa_group_ids)
        self.assertEqual(spec_driven.fa_group_ids, [0, 1])
        self.assertEqual(spec_driven.window_group_ids, legacy.window_group_ids)
        self.assertEqual(spec_driven.window_group_ids, [2, 3])

    def test_spec_driven_matches_legacy_tails(self):
        spec_driven, legacy = self._run_both_modes()
        for gid in sorted(spec_driven.group_metas):
            want = legacy.group_metas[gid]
            got = spec_driven.group_metas[gid]
            self.assertEqual(got, want, f"group[{gid}] meta drifted")
            self.assertEqual(got.tail_tokens, want.tail_tokens)
        # 语义抽查: FA 组 canonical 尾 = hash block;SWA 组 = 窗口;compressor = 窗口-比。
        self.assertEqual(spec_driven.group_metas[0].tail_tokens, 128)
        self.assertEqual(spec_driven.group_metas[1].tail_tokens, 128)
        self.assertEqual(spec_driven.group_metas[2].tail_tokens, 2048)
        self.assertEqual(spec_driven.group_metas[3].tail_tokens, 4096 - 32)

    def test_row_ownership_columns_drive_decision(self):
        groups, compress_ratios = _dsv4_like_groups()
        spec_table = spec_table_builder.build_spec_table(
            groups, layer_compress_ratios=compress_ratios
        )
        conn = _connector(groups, compress_ratios, spec_table)
        conn._init_group_metas()
        for gid, row in enumerate(spec_table.rows):
            is_fa = row.sliding_window is None
            self.assertEqual(
                gid in conn.fa_group_ids,
                is_fa,
                f"group[{gid}] fa 归属与规格表归属轴不一致",
            )
            self.assertEqual(
                gid in conn.window_group_ids, not is_fa
            )

    def test_spec_table_absent_no_regression(self):
        # 无 double-run/authoritative 开关时,旧启发式仍给出同一归属。
        groups, compress_ratios = _dsv4_like_groups()
        conn = _connector(groups, compress_ratios, None)
        conn._init_group_metas()
        self.assertEqual(conn.fa_group_ids, [0, 1])
        self.assertEqual(conn.window_group_ids, [2, 3])


if __name__ == "__main__":
    unittest.main()