"""vLLM/vllm-ascend 0.26.0 统一适配器(补丁收敛 9.1 阶段 4 首个落地)。

覆盖配置 (Qwen3.8-27B / vllm 0.26.0 nightly / davinci3, 见交接文档) 的
全部引擎差异:

- 引擎源码树侧(vLLM scheduler 混合组逐块驱逐等): 由 nightly 镜像合入,
  ``v0260-combined.patch`` 是本仓库的 diff 档案(引擎树已含时零动作,
  校验见 ``VersionAdapter.verify_engine_patches``);
- 运行期注入(vllm-ascend CPU 亲和绑定): ``apply()`` 安装,幂等。

新版本适配流程: 新建 ``v0XY9/adapter.py`` 声明 VLLM_VERSION /
ASCEND_VERSIONS / REQUIRED_ENGINE_PATCHES / apply(),注册表自动发现,
零改动 apply_patch.py。
"""

from __future__ import annotations

VLLM_VERSION = "0.26.0"
ASCEND_VERSIONS = ("0.26.0",)
REQUIRED_ENGINE_PATCHES = ("v0260-combined.patch",)


def apply(ascend_version: str | None = None) -> None:
    """安装 0.26.0 的全部运行期注入(cpu_binding 补丁,幂等)。"""
    from ucm.integration.vllm.patch.v0260.vllm_ascend import cpu_binding_patch

    # module import 即注册 @when_imported 钩子;显式引用避免 linter 优化。
    _ = cpu_binding_patch