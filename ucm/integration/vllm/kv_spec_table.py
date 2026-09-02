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

"""Coordinator (L2) stage-1 primitives: spec table + component voting + checkpoints.

对应《UCM 缓存系统:面向混合模型的分层设计报告》(2026-08) 第 4 章与第 9 章阶段 1:

- ``SpecTable``: 规格表(4.1)。把散落在 Connector 里的模型知识(现状的两份胚胎表:
  ``hla_connector.GroupInfo`` / ``hma_connector.KVCacheGroupMeta``)数据化为
  "每个 kv_cache_group 一行、启动时算一次、运行期只读" 的单表。模型的特殊性只以
  **数据(行)** 存在,不再以 **代码(新 Connector 类)** 存在。
- ``resolve_hit``: 组件投票(4.2)的纯函数。现状雏形在
  ``hla_connector.KVCacheGroupManager.lookup_external_hit_tokens`` 的类内,别的
  Connector 用不上;这里上收为协调器单函数:链式组按组查询取最小(错命比漏命贵),
  对齐到公共刻度(LCM),快照组经检查点目录取最深可用位置。
- ``CheckpointDirectory``: 检查点目录 + 惰性失效(4.3)。键 = (组, 位置, 前缀哈希);
  有效性不存标志、用时现算 -- "最深的 ≤ l 的位置" 即有效,块被淘汰 => 检查点自动
  够不着 => 零通知、零跨层协议,淘汰永远不产生错误命中。

设计纪律(6.5 铁律 / 附录 E5):本模块**零第三方依赖**(不 import vllm / torch /
ucm 包),可在未构建 C++ 扩展的裸环境下直接用 pytest 单测
(见 ``test/suites/Unit/test_kv_spec_table.py``,断言来自 4.5 算例 A/B/C)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "CacheKind",
    "RankRule",
    "RetentionPolicy",
    "SpecRow",
    "SpecTable",
    "CheckpointDirectory",
    "resolve_hit",
    "rank_block_present",
]

# 无检查点可用的哨兵位置:表示"该快照组在任何 ≤ l 的位置都没有可用检查点",
# 引擎对该组状态从 0 开始重推(漏命安全,见 4.3)。
NO_CHECKPOINT = 0


class CacheKind(str, Enum):
    """缓存数据分类轴(6.5 铁律 1:可寻址性是唯一稳定分类轴)。

    CHAIN    链式数据(积木): 普通 KV / MLA 潜向量 / 压缩 KV / 滑动窗口 KV。
              前缀相同即可复用,可拼接。
    SNAPSHOT 快照数据(罐头): Mamba / KDA(记忆编辑状态)/ 循环状态。
              只在精确位置有效,位置不对就废。
    NONE     不缓存: 跨注意力(cross-attention)等。
    """

    CHAIN = "chain"
    SNAPSHOT = "snapshot"
    NONE = "none"


class RankRule(str, Enum):
    """秩规则(4.2): "这一组由哪些 rank 负责写盘、跨 rank 聚合取并集还是交集"。

    ALL_UNION      MLA(内容跨 rank 相同): 任一 rank dump 成功即可,聚合取并集 --
                   用冗余换容错(4.5 算例 B:可用概率 1-p^4)。
    ALL_INTERSECT  非 MLA(各 rank 内容不同): 每个 rank 必须自己 dump,聚合取交集 --
                   以冗余换可用率(4.5 算例 B:可用概率 (1-p)^4)。

    注: ``RetentionPolicy`` 里的检查点间隔是保留策略参数,不是秩规则 -- 两列语义
    不同(见 4.2 秩规则定义段的澄清)。
    """

    ALL_UNION = "all_union"
    ALL_INTERSECT = "all_intersect"


@dataclass(frozen=True)
class RetentionPolicy:
    """快照组的保留策略(4.3): 在哪些位置创建检查点,三触发。

    - 请求结束(仅当有新增计算/新状态;完全命中则跳过);
    - 同一前缀第二次"未被服务"的出现(命中过缓存的不算);
    - 定间隔: 长输入/输出每隔 ``interval`` token 存一个。

    ``grid_alignment``: 检查点位置必须落在的网格粒度(如 mamba2 每 64 token 一格;
    KDA 逐 token 则网格为 1)。位置按网格对齐,与链式块的 block_size、以及
    ``tokens_per_state``(状态压缩粒度)是三个不同的刻度(见 5.1)。
    """

    interval: Optional[int] = None
    grid_alignment: int = 1


@dataclass(frozen=True)
class SpecRow:
    """规格表一行(4.1)。每个 kv_cache_group 一行,启动时算一次、运行期只读。"""

    group_name: str
    kind: CacheKind
    block_size: Optional[int]
    # 每层每 token 字节数(示例: MLA 512B;CSA 的 fp8+scale),仅记账,不参与命中裁决。
    bytes_per_token: Optional[Sequence[int]] = None
    seed: Optional[str] = None
    rank_rule: Optional[RankRule] = None
    retention: Optional[RetentionPolicy] = None

    def __post_init__(self) -> None:
        if self.kind is CacheKind.NONE:
            assert (
                self.block_size is None
            ), f"kind=none 的组 {self.group_name} 不应有 block_size"
        else:
            assert self.block_size is not None and self.block_size > 0, (
                f"组 {self.group_name} 的 block_size 必须为正整数," f"实际 {self.block_size!r}"
            )


class SpecTable:
    """协调器规格表: 模型知识的唯一权威来源(4.1,9.1 动手点①)。"""

    def __init__(self, rows: Iterable[SpecRow]) -> None:
        self.rows: list[SpecRow] = list(rows)
        self._by_name: dict[str, SpecRow] = {r.group_name: r for r in self.rows}
        if len(self._by_name) != len(self.rows):
            raise ValueError("规格表存在重复的组名")
        self.chain_rows = [r for r in self.rows if r.kind is CacheKind.CHAIN]
        self.snapshot_rows = [r for r in self.rows if r.kind is CacheKind.SNAPSHOT]

    def row(self, group_name: str) -> SpecRow:
        return self._by_name[group_name]

    @property
    def lcm_block_size(self) -> int:
        """各组块大小的最小公倍数(LCM): 命中长度对齐的公共刻度(4.2)。"""
        if not self.chain_rows:
            raise ValueError("规格表至少需要一个 chain 组才能计算 LCM(4.2)")
        return math.lcm(*(r.block_size for r in self.chain_rows))

    def __repr__(self) -> str:
        header = f"{'组名':<16}{'kind':<10}{'block':>8}{'秩规则':<14}"
        lines = [header]
        for r in self.rows:
            lines.append(
                f"{r.group_name:<16}{r.kind.value:<10}"
                f"{str(r.block_size):>8}{r.rank_rule.value if r.rank_rule else '-':<14}"
            )
        return "\n".join(lines)


def resolve_hit(
    spec: SpecTable,
    prefix_hashes: Mapping[str, Sequence[bytes]],
    chain_existence: Callable[[SpecRow, Sequence[bytes]], int],
    checkpoints: Mapping[str, "CheckpointDirectory"],
    snapshot_prefix_ids: Optional[Mapping[str, bytes]] = None,
    candidate_L: Optional[int] = None,
) -> tuple[int, int]:
    """组件投票(4.2)的纯函数实现。

    返回 ``(l, p*)``:
    - ``l``: 链式组的公共命中长度 -- 对每个链式组问持久层"最长存在到哪",取最小
      (组件投票 = 取交集,错命比漏命贵),再对齐到各组块大小的公共刻度 LCM。
    - ``p*``: 所有快照组的检查点都就位的最深位置(≤ l)。无快照组时 ``p* = l``;
      任一快照组在 ≤ l 处无检查点,则该组贡献 ``NO_CHECKPOINT(0)``,跨组取最小。

    Args:
        spec: 规格表(链式组与快照组都在其中)。
        prefix_hashes: 按组名给出该请求的块哈希链(组间经独立种子隔离,3.2)。
        chain_existence: ``(row, block_ids) -> tokens``,返回该链式组在持久层里
            从开头连续存在的 token 数(BlockStore[g].lookup_on_prefix 语义)。
        checkpoints: 按组名给出检查点目录。仅快照组会被查询。
        snapshot_prefix_ids: 按组名给出该请求的前缀标识(检查点目录键的一部分);
            缺失的组视为无可用检查点(贡献 0)。
        candidate_L: 引擎本地匹配的候选长度,仅作首轮裁剪(报告伪代码中未参与
            计算,保留参数以待未来接入;当前不改变结果)。
    """
    _ = candidate_L  # 4.2 伪代码中 candidate_L 仅作首轮裁剪,当前不参与计算

    chain_rows = spec.chain_rows
    if not chain_rows:
        raise ValueError("resolve_hit 至少需要一个 chain 组(4.2)")

    l: int = math.inf
    for g in chain_rows:
        block_ids = prefix_hashes.get(g.group_name, ())
        l_g = max(int(chain_existence(g, list(block_ids))), 0)
        l = min(l, l_g)

    lcm = spec.lcm_block_size
    l = (l // lcm) * lcm

    # 快照组: 跨组取最小(所有组的检查点都就位才可跳过 [0, p*))。
    if not spec.snapshot_rows:
        return l, l

    p_star: int = math.inf
    for s in spec.snapshot_rows:
        prefix_id = (
            snapshot_prefix_ids.get(s.group_name) if snapshot_prefix_ids else None
        )
        if prefix_id is None:
            p_s = NO_CHECKPOINT
        else:
            p_s = checkpoints[s.group_name].deepest_candidate(l, prefix_id)
        p_star = min(p_star, p_s)
    return l, p_star


def rank_block_present(rule: RankRule, dump_success: Sequence[bool]) -> bool:
    """按秩规则聚合各 rank 的 dump 结果(4.5 算例 B 的判定语义)。

    - ``ALL_UNION``: 任一 rank dump 成功即可(4 份冗余换容错);
    - ``ALL_INTERSECT``: 每个 rank 都必须成功(缺一片即该块不可用)。
    """
    if rule is RankRule.ALL_UNION:
        return any(dump_success)
    if rule is RankRule.ALL_INTERSECT:
        return all(dump_success)
    raise ValueError(f"未知秩规则: {rule!r}")


def deepest_snapshot_p_star(
    directories: Mapping[str, "CheckpointDirectory"],
    prefix_at_position: Callable[[int], Optional[bytes]],
    num_computed_tokens: int,
    total_hit_tokens: int,
    lcm_block_size: int,
) -> int:
    """检查点目录驱动的快照 p*(4.3)纯函数: "最深的 ≤ l 且前缀链匹配"。

    检查点键 = (组, 位置, 前缀哈希),且前缀哈希随位置链式变化: 位置 p 的检查点
    只对"前缀哈希链匹配到 p"的请求可见。因此从链式候选 ``total_hit_tokens``
    向下逐 LCM 边界扫描,每个候选位置用该位置自己的链式前缀哈希
    (``prefix_at_position(p)``,由调用方按主链式组派生)查询**所有**快照组目录;
    所有组在同一个 p 都登记过,该 p 才可用(跨组取最小的语义,4.2 ④′),返回
    最深者。无快照目录 / 无可用位置时返回 ``num_computed_tokens``(状态重推,
    漏命安全)。

    惰性失效天然成立: 链式块被淘汰 => ``prefix_at_position(p)`` 对应的链或
    目录项不再完整 => 查不到 => 自动够不着,零通知零跨层协议(4.3)。

    Args:
        directories: 按快照组名 -> 检查点目录(鸭子类型 ``positions(prefix)``)。
        prefix_at_position: 位置 -> 该位置的前缀哈希(链式,随位置变化)。
        num_computed_tokens: 引擎本地已算 token 数(下界)。
        total_hit_tokens: 链式候选绝对位置(上界,即 l)。
        lcm_block_size: 快照边界网格(与链式 LCM 对齐)。
    """
    if not directories:
        return total_hit_tokens
    for p in range(total_hit_tokens, num_computed_tokens - 1, -lcm_block_size):
        if p <= num_computed_tokens:
            continue
        prefix_hash = prefix_at_position(p)
        if not prefix_hash:
            continue
        all_groups_ready = True
        for directory in directories.values():
            if p not in directory.positions(prefix_hash):
                all_groups_ready = False
                break
        if all_groups_ready:
            return p
    return num_computed_tokens


@dataclass
class CheckpointDirectory:
    """快照组检查点目录(4.3)。

    键 = (组, 位置, 前缀哈希): 只有与当前请求前缀匹配的目录项才参与 p* 候选,
    "位置对、内容错" 的跨前缀错命不会发生。

    有效性**惰性**(不存有效标志,用时现算): ``deepest_candidate(l, ...)`` 天然
    = "链式块最长存在到哪",目录里 "最深的 ≤ l 的位置" 即有效。块被淘汰 => 检查点
    自动够不着 => 零通知、零跨层协议。快照条目自身被存储淘汰 = 下次 Get miss ->
    调用方 ``on_get_miss`` 将条目作废,引擎退化为"该段状态重推"(漏命安全)。
    """

    group_name: str
    grid_alignment: int = 1
    _positions_by_prefix: dict[bytes, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.grid_alignment >= 1

    def _align(self, position: int) -> int:
        """检查点位置向下对齐到网格((组,位置,前缀哈希)里的"位置"刻度,5.1)。"""
        return (position // self.grid_alignment) * self.grid_alignment

    def register(self, position: int, prefix_hash: bytes) -> None:
        """Put 完成后登记 (组, 位置, 前缀哈希)。重复登记是幂等的。"""
        self._positions_by_prefix.setdefault(prefix_hash, set()).add(
            self._align(position)
        )

    def positions(self, prefix_hash: bytes) -> set[int]:
        return set(self._positions_by_prefix.get(prefix_hash, set()))

    def deepest_candidate(self, l: int, prefix_hash: bytes) -> int:
        """惰性失效的核心: 返回 ≤ l 的最深已登记检查点;没有则 NO_CHECKPOINT。"""
        best = NO_CHECKPOINT
        for pos in self._positions_by_prefix.get(prefix_hash, ()):
            if pos <= l and pos > best:
                best = pos
        return best

    def on_get_miss(self, position: int, prefix_hash: bytes) -> None:
        """Get miss(条目被存储淘汰)-> 该目录项自然作废(4.3。幂等)。"""
        aligned = self._align(position)
        positions = self._positions_by_prefix.get(prefix_hash)
        if positions is None:
            return
        positions.discard(aligned)

    def on_request_ended(
        self,
        prefix_hash: bytes,
        end_position: int,
        *,
        interval: Optional[int] = None,
    ) -> list[int]:
        """保留策略触发①+③: 请求结束 + 定间隔,登记新增检查点。

        只登记 ``(上次已有位置, end_position]`` 内的定间隔点与 ``end_position``;
        完全命中、无新增内容的请求由调用方跳过本方法。返回本次新建的位置列表。
        """
        positions = self._positions_by_prefix.setdefault(prefix_hash, set())
        created: list[int] = []
        step = interval if interval is not None else self.grid_alignment
        assert step >= 1
        for pos in range(step, end_position + 1, step):
            pos = self._align(pos)
            if pos > 0 and pos not in positions:
                positions.add(pos)
                created.append(pos)
        end = self._align(end_position)
        if end > 0 and end not in positions:
            positions.add(end)
            created.append(end)
        return created

    def on_unserved_seen(self, prefix_hash: bytes, position: int) -> bool:
        """保留策略触发②: 同一前缀第二次"未被服务"-> 在公共边界存检查点。

        "未被服务"指该边界从未被缓存命中服务过(命中过缓存的不算,它已享受)。
        返回本次是否新建了检查点。
        """
        aligned = self._align(position)
        counter = self._unserved.setdefault((prefix_hash, aligned), 0)
        counter += 1
        self._unserved[(prefix_hash, aligned)] = counter
        if counter == 2:
            self.register(aligned, prefix_hash)
            return True
        return False

    # (前缀哈希, 边界位置) -> 该边界被"未见服务"的次数;目录内的簿记状态。
    _unserved: dict[tuple[bytes, int], int] = field(default_factory=dict)
