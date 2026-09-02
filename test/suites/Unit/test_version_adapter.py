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

"""补丁收敛(9.1 阶段 4): '每版本一适配器' 注册表单测。

零依赖模式(同 test_kv_spec_table): 用临时目录 + 预置 fake 包命名空间
验证注册表自动发现 / 查表 / 档案缺失告警 / 未迁移目录跳过。
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "patch" / "version_adapter.py"


def _load_pure_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fake_adapter_module(version, *, patch_files=(), ascend_versions=()):
    mod = types.ModuleType("_fake_adapter")
    mod.VLLM_VERSION = version
    mod.ASCEND_VERSIONS = ascend_versions or (version,)
    mod.REQUIRED_ENGINE_PATCHES = tuple(patch_files)

    def apply():
        apply.called = True

    mod.apply = apply
    return mod


version_adapter = _load_pure_module("version_adapter", ADAPTER_PATH)


class VersionAdapterRegistryTest(unittest.TestCase):
    """注册表自动发现与查表。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = self._tmp.name
        self._fake_pkg = "test.vllm.patch"
        self._saved = {
            name: sys.modules.pop(name, None)
            for name in sorted(list(sys.modules))
            if name.startswith(self._fake_pkg)
        }

    def tearDown(self):
        for name, mod in self._saved.items():
            if mod is not None:
                sys.modules[name] = mod
        self._tmp.cleanup()

    def _install_adapter(self, version_dir, adapter_mod):
        os.makedirs(os.path.join(self._dir, version_dir), exist_ok=True)
        name = f"{self._fake_pkg}.{version_dir}.adapter"
        sys.modules[name] = adapter_mod

    def test_discover_and_lookup(self):
        self._install_adapter("v0260", _fake_adapter_module("0.26.0"))
        table = version_adapter.discover_adapters(
            package=self._fake_pkg, directories=[self._dir]
        )
        self.assertEqual(list(table.keys()), ["0.26.0"])
        adapter = version_adapter.get_adapter("0.26.0", table)
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.vllm_version, "0.26.0")
        self.assertIsNone(version_adapter.get_adapter("0.99.0", table))

    def test_apply_registered_and_idempotent_calls(self):
        mod = _fake_adapter_module("0.26.0")
        self._install_adapter("v0260", mod)
        table = version_adapter.discover_adapters(
            package=self._fake_pkg, directories=[self._dir]
        )
        adapter = version_adapter.get_adapter("0.26.0", table)
        adapter.apply()
        self.assertTrue(mod.apply.called)

    def test_unmigrated_version_dir_skipped(self):
        # 只有实现文件、没有 adapter.py 的版本目录(未迁移)应被跳过。
        os.makedirs(os.path.join(self._dir, "v0110", "vllm"), exist_ok=True)
        Path(os.path.join(self._dir, "v0110", "vllm", "pc_patch.py")).write_text(
            "pass\n"
        )
        self._install_adapter("v0260", _fake_adapter_module("0.26.0"))
        table = version_adapter.discover_adapters(
            package=self._fake_pkg, directories=[self._dir]
        )
        self.assertEqual(list(table.keys()), ["0.26.0"])

    def test_missing_engine_patch_archive_reported(self):
        mod = _fake_adapter_module("0.26.0", patch_files=("v0260-combined.patch",))
        self._install_adapter("v0260", mod)
        table = version_adapter.discover_adapters(
            package=self._fake_pkg, directories=[self._dir]
        )
        adapter = version_adapter.get_adapter("0.26.0", table)
        # 用假目录做档案根: 假 v0260 目录下没有该 patch 文件 => 缺失清单非空。
        missing = adapter.verify_engine_patches(base_dir=self._dir)
        self.assertIn("v0260-combined.patch", missing)

    def test_verify_engine_patches_ok_when_archive_present(self):
        mod = _fake_adapter_module("0.26.0", patch_files=("v0260-combined.patch",))
        self._install_adapter("v0260", mod)
        os.makedirs(os.path.join(self._dir, "v0260"), exist_ok=True)
        Path(os.path.join(self._dir, "v0260", "v0260-combined.patch")).write_text(
            "diff --git a/x b/x\n"
        )
        table = version_adapter.discover_adapters(
            package=self._fake_pkg, directories=[self._dir]
        )
        self.assertEqual(
            version_adapter.get_adapter("0.26.0", table).verify_engine_patches(
                base_dir=self._dir
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()