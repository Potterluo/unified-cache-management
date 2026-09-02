"""Version-specific patches for vLLM-Ascend 0.26.0.

补丁收敛(9.1 阶段 4)后,本目录由 ``adapter.py`` 统一描述(版本声明 +
引擎 diff 档案校验 + 运行期注入入口);子目录保留历史实现文件。
"""

from ucm.integration.vllm.patch.v0260.adapter import apply  # noqa: F401
