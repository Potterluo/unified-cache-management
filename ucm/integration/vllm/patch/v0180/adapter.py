"""v0180 版本适配器(补丁收敛 9.1 阶段 4)。

vLLM/vllm-ascend 0.18.0(+0.17.0 别名): pc(prefix cache);0.17.0 仅 UCMConnector 指标别名。
"""

from __future__ import annotations

VLLM_VERSION = "0.18.0"
ASCEND_VERSIONS = ("0.18.0",)
REQUIRED_ENGINE_PATCHES = ()
ASCEND_VERSIONS = ("0.18.0", "0.17.0")
ALIASES = ("0.17.0",)

def apply(ascend_version: str | None = None) -> None:
    """安装 v0180 版运行期注入(import 即注册钩子,幂等)。"""
    if ascend_version == "0.17.0":
        # vllm-ascend 0.17.0 复用本适配器: 仅注册 UCMConnector 指标别名。
        from ucm.integration.vllm.patch.v0180.vllm_ascend import (  # noqa: F401
            ucm_connector_patch,
        )

        _ = ucm_connector_patch
        return
    from ucm.integration.vllm.patch.v0180.vllm import pc_patch  # noqa: F401
    from ucm.integration.vllm.patch.v0180.vllm_ascend import (  # noqa: F401
        pc_ascend_patch,
    )

    _ = pc_patch, pc_ascend_patch
