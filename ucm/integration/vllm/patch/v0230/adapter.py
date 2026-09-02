"""v0230 版本适配器(补丁收敛 9.1 阶段 4)。

vLLM/vllm-ascend 0.23.0: hybrid cache 恢复 + CPU 亲和绑定 + SFA KV 传输。
"""

from __future__ import annotations

VLLM_VERSION = "0.23.0"
ASCEND_VERSIONS = ("0.23.0",)
REQUIRED_ENGINE_PATCHES = ()

def apply(ascend_version: str | None = None) -> None:
    """安装 v0230 版运行期注入(import 即注册钩子,幂等)。"""
    from ucm.integration.vllm.patch.v0230.vllm_ascend import (  # noqa: F401
        ascend_hybrid_cache_patch,
        cpu_binding_patch,
        sfa_kv_transfer_patch,
    )

    _ = ascend_hybrid_cache_patch, cpu_binding_patch, sfa_kv_transfer_patch
