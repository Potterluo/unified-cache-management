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

"""Unit tests for UCM patch-version selection (apply_patch.py).

apply_patch imports ``ucm.logger`` (needs the built extension), so the ``ucm``
package is stubbed here and the module is loaded directly from its source file
(same pattern as test_spec_table_builder).
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
APPLY_PATCH_PATH = (
    REPO_ROOT / "ucm" / "integration" / "vllm" / "patch" / "apply_patch.py"
)


def _load_apply_patch():
    ucm = types.ModuleType("ucm")
    ucm.__path__ = []
    logger = types.ModuleType("ucm.logger")

    def init_logger(*_args, **_kwargs):
        import logging

        return logging.getLogger("ucm")

    logger.init_logger = init_logger
    sys.modules["ucm"] = ucm
    sys.modules["ucm.logger"] = logger
    spec = importlib.util.spec_from_file_location("apply_patch", APPLY_PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["apply_patch"] = module
    spec.loader.exec_module(module)
    return module


apply_patch = _load_apply_patch()


class VersionNormalizationTest(unittest.TestCase):
    def test_strip_build_metadata(self):
        # nightly / build metadata: 0.26.0+empty -> 0.26.0
        self.assertEqual(apply_patch._strip_build("0.26.0+empty"), "0.26.0")
        self.assertEqual(apply_patch._strip_build("0.28.0+empty"), "0.28.0")
        self.assertEqual(apply_patch._strip_build(None), None)

    def test_norm_version(self):
        self.assertEqual(apply_patch._norm_version("0.18.0rc1"), "0.18.0")
        self.assertEqual(apply_patch._norm_version("0.28.0"), "0.28.0")
        self.assertEqual(apply_patch._norm_version("0.11.0.post1"), "0.11.0")

    def test_supported_versions_include_latest(self):
        versions = apply_patch.get_supported_versions()
        for v in ("0.26.0", "0.27.0", "0.28.0"):
            self.assertIn(v, versions)


if __name__ == "__main__":
    unittest.main()
