"""v0270 版本适配器(补丁收敛 9.1 阶段 4)。

vLLM 0.27.0: 无需版本专属运行期补丁(上移引擎),apply 为空声明;kimi_k3 钩子为公共补丁。
"""

from __future__ import annotations

VLLM_VERSION = "0.27.0"
ASCEND_VERSIONS = ("0.27.0",)
REQUIRED_ENGINE_PATCHES = ()
from ucm.logger import init_logger

def apply(ascend_version: str | None = None) -> None:
    """安装 v0270 版运行期注入(import 即注册钩子,幂等)。"""
    # 0.27.0 无需版本专属补丁: hybrid prefix-cache fix 已上移引擎
    # (与 apply_patch.py 0.27.0 分支的既有注释一致);kimi_k3 钩子属于
    # 公共补丁,由 apply_all_patches 直接注册。
    init_logger(__name__).info(
        "UCM v0270 adapter: no version-specific runtime patches needed"
    )
