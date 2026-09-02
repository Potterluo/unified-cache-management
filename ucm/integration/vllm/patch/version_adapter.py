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

"""补丁收敛(9.1 阶段 4): '每版本一适配器' 的协议与注册表。

目标形态: 每个引擎版本一个 ``适配器``(v0XYZ/adapter.py),集中声明该版本
的全部引擎差异(源码树 diff 档案校验 + 运行时注入入口 + 特殊适配);版本分派
由注册表自动发现驱动,不再维护 ``apply_patch.py`` 里的巨型 match。

``VersionAdapter`` 协议:
- ``vllm_version``: 本适配器对应的 vLLM 版本(主键);
- ``ascend_versions``: 可适配的 vllm-ascend 版本(通常与 vLLM 同号;
  个别版本错位如 0.17.0 复用 v0180 适配器时在此登记);
- ``required_engine_patches``: 引擎源码树必须已包含的 diff 档案名
  (如 ``v0260-combined.patch``);``verify_engine_patches`` 在运行期校验
  并仅告警(引擎树已含时零动作),把"档案"变成主动契约;
- ``apply()``: 运行时注入入口(安装该版本全部 monkey-patch,幂等)。

注册表从包目录自动发现(扫描 v0XYZ/adapter.py,读取模块级常量),新版本
只需新增 ``v0XY9/adapter.py``,零改动 ``apply_patch.py`` 的版本分派。

本模块零第三方依赖(同 ``kv_spec_table``),可裸环境直接单测。
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

__all__ = [
    "VersionAdapter",
    "discover_adapters",
    "get_adapter",
]

# 版本目录统一前缀(与现有 v0XYZ 布局一致)。
_VERSION_DIR_PREFIX = "v0"
# 每个版本目录内的适配器模块名。
_ADAPTER_MODULE = "adapter"


@dataclass(frozen=True)
class VersionAdapter:
    """一个引擎版本的统一适配器(9.1 阶段 4 的收敛单元)。"""

    vllm_version: str
    apply: Callable[[], None]
    ascend_versions: tuple[str, ...] = ()
    required_engine_patches: tuple[str, ...] = ()
    module_path: str = ""

    def verify_engine_patches(self, base_dir: Optional[str] = None) -> list[str]:
        """校验引擎源码树应已包含的 diff 档案,返回缺失清单(仅告警用)。

        档案在版本目录内(``v0XYZ`` = 版本号去点,如 0.26.0 -> v0260);
        diff 内容属引擎侧合入状态,由引擎仓库 git 历史担保,这里只校验
        档案文件存在可寻址。``base_dir`` 可注入(单测用假目录),默认
        本模块所在目录。
        """
        missing: list[str] = []
        base = base_dir or os.path.dirname(os.path.abspath(__file__))
        version_dir = "v" + self.vllm_version.replace(".", "")
        for name in self.required_engine_patches:
            path = os.path.join(base, version_dir, name)
            if not os.path.isfile(path):
                missing.append(name)
        return missing


def discover_adapters(
    package: Optional[str] = None,
    directories: Optional[Sequence[str]] = None,
) -> dict[str, VersionAdapter]:
    """扫描 v0XYZ/adapter.py,构建 {vllm_version: VersionAdapter} 注册表。

    延迟 import 各适配器模块(仅加载声明头),不触发引擎注入。
    ``directories``: 扫描目录(默认本模块所在目录),单测可注入临时目录;
    适配器模块经标准 import 解析(测试可预置 fake 包命名空间)。
    """
    base_package = package or __name__.rsplit(".", 1)[0]
    search_dirs = (
        list(directories)
        if directories is not None
        else [os.path.dirname(os.path.abspath(__file__))]
    )
    adapters: dict[str, VersionAdapter] = {}
    for package_dir in search_dirs:
        if not os.path.isdir(package_dir):
            continue
        for name in sorted(os.listdir(package_dir)):
            if not name.startswith(_VERSION_DIR_PREFIX):
                continue
            adapter_module_name = f"{base_package}.{name}.{_ADAPTER_MODULE}"
            try:
                mod = importlib.import_module(adapter_module_name)
            except ImportError:
                # 目录存在但尚无 adapter.py: 尚未迁移的版本目录,跳过(可机械迁移)。
                continue
            vllm_version = getattr(mod, "VLLM_VERSION", None)
            apply_fn = getattr(mod, "apply", None)
            if not vllm_version or not callable(apply_fn):
                continue
            adapters[vllm_version] = VersionAdapter(
                vllm_version=vllm_version,
                apply=apply_fn,
                ascend_versions=tuple(getattr(mod, "ASCEND_VERSIONS", ())),
                required_engine_patches=tuple(
                    getattr(mod, "REQUIRED_ENGINE_PATCHES", ())
                ),
                module_path=adapter_module_name,
            )
    return adapters


def get_adapter(
    version: str,
    adapters: Optional[dict[str, VersionAdapter]] = None,
) -> Optional[VersionAdapter]:
    """按归一化 vLLM 版本查适配器;无适配器返回 None(版本未适配,不报错)。"""
    table = adapters if adapters is not None else discover_adapters()
    return table.get(version)