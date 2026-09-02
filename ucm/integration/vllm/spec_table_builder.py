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

"""规格表构建与双跑记账(4.4 C1 的 "双跑:旧逻辑为准,新表只记账比对")。

现状里模型知识散在多处(``hla_connector.GroupInfo``、``hma_connector.
KVCacheGroupMeta``),实现本报告的规格表时要先在**不改变旧逻辑**的前提下把
旧逻辑跑出来的组信息与规格表逐组比对:一致才说明映射正确,不一致立即告警并
冻结切新。本模块提供两件事:

- ``build_spec_table``: 从 vLLM 的 ``KVCacheConfig.kv_cache_groups`` 构建
  :class:`~ucm.integration.vllm.kv_spec_table.SpecTable`(启动时算一次);
- ``double_run_ledger``: 把旧 ``GroupInfo`` 列表与规格表逐组记账比对,
  返回不一致清单(由调用方告警/记指标)。

注意:本模块只在 Connector 内被引用(运行期),因此允许 import vllm / ucm.logger;
与零依赖的 ``kv_spec_table.py`` 保持模块边界。
"""

import math
import os
from typing import Callable, Iterable, Optional, Sequence

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from ucm.integration.vllm.kv_spec_table import (
    CacheKind,
    RankRule,
    RetentionPolicy,
    SpecRow,
    SpecTable,
)

# 当 UCM_SPEC_TABLE_DOUBLE_RUN 开启时执行双跑记账(默认关闭,零行为影响)。
_DOUBLE_RUN_FLAG = "UCM_SPEC_TABLE_DOUBLE_RUN"
# 当 UCM_SPEC_TABLE_AUTHORITATIVE 开启时,resolve_hit 成为链式命中主裁决,
# 旧逻辑转 shadow 记账(4.4 C1 双跑对齐后方可开启;默认关闭 = 旧逻辑为准)。
_AUTHORITATIVE_FLAG = "UCM_SPEC_TABLE_AUTHORITATIVE"


def spec_table_double_run_enabled() -> bool:
    """双跑开关: ``UCM_SPEC_TABLE_DOUBLE_RUN`` 为 1/true/yes/on 时开启。"""
    value = os.getenv(_DOUBLE_RUN_FLAG, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def spec_table_authoritative_enabled() -> bool:
    """切新开关: ``UCM_SPEC_TABLE_AUTHORITATIVE`` 为 1/true/yes/on 时开启。

    开启后 ``resolve_hit``(规格表组件投票 + 检查点目录)成为链式命中裁决的
    主路径,旧 ``_lookup_external_hit_tokens_legacy`` 的链式候选转 shadow 记账
    (不一致仍告警,但执行以新逻辑为准)。快照 p* 仍走旧 reverse scan(阶段 2
    SnapshotStore + 检查点目录落地后一并切换)。
    """
    value = os.getenv(_AUTHORITATIVE_FLAG, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def block_size_from_spec(spec: KVCacheSpec) -> Optional[int]:
    """组块大小: 统一规格取首个成员 spec,普通 spec 直接取 block_size。"""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        for s in spec.kv_cache_specs.values():
            return s.block_size
        return None
    return spec.block_size


def compress_ratio_from_spec(spec: KVCacheSpec) -> int:
    """压缩比: 统一规格取首个成员 spec,普通 spec 取 ``compress_ratio``(缺省 1)。"""
    if isinstance(spec, UniformTypeKVCacheSpecs):
        for s in spec.kv_cache_specs.values():
            return compress_ratio_from_spec(s)
        return 1
    return int(getattr(spec, "compress_ratio", 1) or 1)


def logical_block_size_from_spec(spec: KVCacheSpec) -> Optional[int]:
    """规格表 block 列的语义(4.1): 一个缓存块装多少个 **token**。

    Ascend 压缩组(如 DS-V4 的 C4/C128)的引擎 ``spec.block_size`` 是存储刻度
    (``storage_block_size = block_size // compress_ratio``,见 6.2),而 FAWA
    旧表 ``token_block_size = block_size * compress_ratio`` 才是逻辑 token
    刻度;规格表必须用逻辑刻度才能与旧逻辑对齐(4.4 C1 双跑记账)。
    """
    base = block_size_from_spec(spec)
    if base is None:
        return None
    return base * compress_ratio_from_spec(spec)


def is_mamba_align_spec(spec: KVCacheSpec) -> bool:
    """mamba-align 快照组判断(状态 = 块对齐快照,见 2.2/5.2)。

    UniformType 组的成员可混合(如 MLA + mamba 同组):只要存在 mamba(align)
    层,该组就是快照组(引擎按组规格管理状态,见 2.1)。
    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return any(is_mamba_align_spec(s) for s in spec.kv_cache_specs.values())
    return isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"


def is_mla_spec(spec: KVCacheSpec) -> bool:
    """MLA 组判断: 内容跨 rank 相同(共享池去重成立),秩规则 = all_union。

    非 MLA 组各 rank 哈希盐含 rank 编号、各存各的(4.5 算例 B 前提)。
    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return any(is_mla_spec(s) for s in spec.kv_cache_specs.values())
    return isinstance(spec, MLAAttentionSpec)


def _group_tag(spec: KVCacheSpec) -> str:
    if is_mla_spec(spec):
        return "mla"
    if is_mamba_align_spec(spec):
        return "mamba"
    if isinstance(spec, FullAttentionSpec):
        return "fa"
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return "mix"
    return "group"


def spec_kind(spec: KVCacheSpec) -> CacheKind:
    """4.1 的 kind 列: 快照(状态)组 / 链式组 / 不缓存。"""
    if is_mamba_align_spec(spec):
        return CacheKind.SNAPSHOT
    if block_size_from_spec(spec) is None:
        return CacheKind.NONE
    return CacheKind.CHAIN


def legacy_chain_candidate_l(
    num_computed_tokens: int,
    full_attn_group_ids: Sequence[int],
    group_block_ids: Sequence[Sequence[bytes]],
    lookup_on_prefix: Callable[[Sequence[bytes]], int],
    block_sizes: Sequence[int],
    lcm_block_size: int,
) -> int:
    """复算旧 HLA 逻辑的链式命中候选长度(4.4 C1 双跑记账的对照基准)。

    与 ``hla_connector.KVCacheGroupManager`` 旧逻辑的 Stage-1(full-attn 组
    lookup_on_prefix 取 min、向下对齐 LCM)做等价回归,是 ``resolve_hit`` 切新前
    必须对齐的数字。纯函数、零依赖,供双跑记账与单测共用同一基准。

    Returns:
        旧逻辑的链式候选 ``l``(绝对 token 位置,已对齐 LCM,≤ 所有 full-attn
        组的最长存在)。注意: 这是**链式候选**,不含快照状态检查(Stage-2 的
        lookup_on_reverse 属快照语义,阶段 2 SnapshotStore 落地后单独回归)。
    """
    assert len(full_attn_group_ids) == len(block_sizes)
    candidates: list[int] = []
    for gid, block_size in zip(full_attn_group_ids, block_sizes):
        fa_block_ids = group_block_ids[gid]
        fa_hbm_blocks = num_computed_tokens // block_size
        fa_external = fa_block_ids[fa_hbm_blocks:]
        if not fa_external:
            # 该组没有可查的外部块: 视为命中 = 0(不能算进候选)。
            candidates.append(0)
            continue
        try:
            fa_hit_blocks = lookup_on_prefix(fa_external) + 1
        except Exception:
            # 与旧逻辑一致: lookup 异常按 miss 处理,不升级为错命。
            candidates.append(0)
            continue
        candidates.append(max(fa_hit_blocks, 0) * block_size)
    min_external_hit_tokens = min(candidates)
    external_hit_tokens = (
        min_external_hit_tokens // lcm_block_size
    ) * lcm_block_size
    return num_computed_tokens + external_hit_tokens


def build_spec_table(
    kv_cache_groups: Sequence,
    *,
    group_seeds: Optional[Sequence[str]] = None,
) -> SpecTable:
    """从 vLLM ``KVCacheConfig.kv_cache_groups`` 构建规格表(4.1)。

    ``group_seeds``: 可选的每组哈希种子(与旧逻辑同源,记账比对用);不传则由
    ``_group_tag`` 派生稳定标签。
    """
    rows: list[SpecRow] = []
    for group_id, group in enumerate(kv_cache_groups):
        spec = group.kv_cache_spec
        kind = spec_kind(spec)
        # 块大小用逻辑 token 刻度(4.1: 一个缓存块装多少个 token):Ascend 压缩组
        # 的引擎 spec.block_size 是 storage 刻度,须乘 compress_ratio 与 FAWA
        # 旧表 token_block_size 对齐(6.2 / 4.4 C1)。
        block_size = logical_block_size_from_spec(spec)
        rank_rule = RankRule.ALL_UNION if is_mla_spec(spec) else RankRule.ALL_INTERSECT
        retention = (
            RetentionPolicy(grid_alignment=block_size or 1)
            if kind is CacheKind.SNAPSHOT
            else None
        )
        seed = (
            group_seeds[group_id]
            if group_seeds and group_id < len(group_seeds)
            else None
        )
        rows.append(
            SpecRow(
                group_name=f"{_group_tag(spec)}{group_id}",
                kind=kind,
                block_size=block_size,
                bytes_per_token=None,
                seed=seed,
                rank_rule=rank_rule,
                retention=retention,
            )
        )
    return SpecTable(rows)


def double_run_ledger(
    spec: SpecTable,
    legacy_groups: Iterable,
) -> list[str]:
    """双跑记账: 逐组比对旧组信息与规格表,返回不一致描述清单。

    旧逻辑为准 -- 本函数只读比对,不产生任何行为变更;调用方负责对非空清单
    告警(``logger.warning``)并记录指标。兼容两份旧的胚胎表(4.4 C1 / 9.1 动手点①):

    - ``hla_connector.GroupInfo``(字段 block_size / is_mamba_align / layer_names);
    - ``hma_connector.KVCacheGroupMeta``(字段 token_block_size,无 mamba 标记)。

    比对维度:

    - 组数量一致;
    - 每组的 block 大小一致(GroupInfo.block_size / KVCacheGroupMeta.token_block_size);
    - kind 一致(仅当旧表声明了 ``is_mamba_align`` 字段时比对;Fast 表不表达
      快照语义,跳过该项以免误报)。
    """
    mismatches: list[str] = []
    legacy = list(legacy_groups)

    if len(legacy) != len(spec.rows):
        mismatches.append(
            f"组数量不一致: legacy={len(legacy)} vs spec_table={len(spec.rows)}"
        )

    for info in legacy:
        group_id = getattr(info, "group_id", None)
        if group_id is None or group_id >= len(spec.rows):
            mismatches.append(f"legacy GroupInfo(group_id={group_id}) 超出规格表范围")
            continue
        row = spec.rows[group_id]
        legacy_bs = getattr(info, "block_size", None)
        if legacy_bs is None:
            # hma_connector.KVCacheGroupMeta 用 token_block_size 表达块大小。
            legacy_bs = getattr(info, "token_block_size", None)
        if legacy_bs is not None and legacy_bs != row.block_size:
            mismatches.append(
                f"group[{group_id}] block_size: legacy={legacy_bs} vs "
                f"spec_table={row.block_size}"
            )
        legacy_mamba = getattr(info, "is_mamba_align", None)
        if legacy_mamba is not None and legacy_mamba != (
            row.kind is CacheKind.SNAPSHOT
        ):
            mismatches.append(
                f"group[{group_id}] kind: legacy.is_mamba_align={legacy_mamba} vs "
                f"spec_table={row.kind.value}"
            )
    return mismatches
