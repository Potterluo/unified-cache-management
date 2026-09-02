"""v0202 版本适配器(补丁收敛 9.1 阶段 4)。

vLLM/vllm-ascend 0.20.2: Ascend hybrid cache 恢复(前缀缓存对齐) + CPU 亲和绑定。
"""

from __future__ import annotations

VLLM_VERSION = "0.20.2"
ASCEND_VERSIONS = ("0.20.2",)
REQUIRED_ENGINE_PATCHES = ()

def apply(ascend_version: str | None = None) -> None:
    """安装 v0202 版运行期注入(import 即注册钩子,幂等)。"""
    from ucm.integration.vllm.patch.v0202.vllm_ascend import (  # noqa: F401
        ascend_hybrid_cache_patch,
        cpu_binding_patch,
    )

    _ = ascend_hybrid_cache_patch, cpu_binding_patch
