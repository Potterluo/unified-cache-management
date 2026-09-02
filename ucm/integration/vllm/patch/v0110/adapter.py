"""v0110 版本适配器(补丁收敛 9.1 阶段 4)。

vLLM/vllm-ascend 0.11.0: pc(prefix cache) + 可选 sparse(ENABLE_SPARSE=1)。
"""

from __future__ import annotations

VLLM_VERSION = "0.11.0"
ASCEND_VERSIONS = ("0.11.0",)
REQUIRED_ENGINE_PATCHES = (
    "vllm-adapt.patch",
    "vllm-adapt-rerope.patch",
    "vllm-adapt-sparse.patch",
    "vllm-ascend-adapt.patch",
)

def apply(ascend_version: str | None = None) -> None:
    """安装 v0110 版运行期注入(import 即注册钩子,幂等)。"""
    import os

    _sparse = os.getenv("ENABLE_SPARSE", "0").lower() in ("1", "true", "yes", "on")
    from ucm.integration.vllm.patch.v0110.vllm import pc_patch  # noqa: F401
    from ucm.integration.vllm.patch.v0110.vllm_ascend import (  # noqa: F401
        pc_ascend_patch,
    )

    _ = pc_patch, pc_ascend_patch
    if _sparse:
        from ucm.integration.vllm.patch.v0110.vllm import sparse_patch  # noqa: F401
        from ucm.integration.vllm.patch.v0110.vllm_ascend import (  # noqa: F401
            sparse_ascend_patch,
        )

        _ = sparse_patch, sparse_ascend_patch
