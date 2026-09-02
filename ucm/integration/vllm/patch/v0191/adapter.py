"""v0191 版本适配器(补丁收敛 9.1 阶段 4)。

vLLM/vllm-ascend 0.19.1: pc(prefix cache) + CPU 亲和绑定。
"""

from __future__ import annotations

VLLM_VERSION = "0.19.1"
ASCEND_VERSIONS = ("0.19.1",)
REQUIRED_ENGINE_PATCHES = ()

def apply(ascend_version: str | None = None) -> None:
    """安装 v0191 版运行期注入(import 即注册钩子,幂等)。"""
    from ucm.integration.vllm.patch.v0191.vllm import pc_patch  # noqa: F401
    from ucm.integration.vllm.patch.v0191.vllm_ascend import (  # noqa: F401
        cpu_binding_patch,
        pc_ascend_patch,
    )

    _ = pc_patch, cpu_binding_patch, pc_ascend_patch
