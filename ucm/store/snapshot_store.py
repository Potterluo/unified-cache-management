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

"""Stage-2 primitive: SnapshotStore (position-keyed snapshot storage).

对应《UCM 缓存系统:面向混合模型的分层设计报告》(2026-08) 第 5 章与 9.1 阶段 2:

- SnapshotStore 存储**快照数据**(Mamba / KDA / 循环状态 -- "罐头"): 只在精确位置
  有效,位置不对就废。键 = ``(组, 位置, 前缀哈希)``(5.1);
- 操作: ``Put``(写,首次提交获胜做去重) / ``Get``(= CoW: 取用前复制防污染,
  report 5.1) / ``Donate``(零拷贝移交写方缓冲所有权) / ``Touch``(热度脉冲,
  存储无会话语义,只收瞬时信号,6.4);
- ``SnapshotGroup``: 快照组组合件 = SnapshotStore + 检查点目录(4.3 惰性失效)。
  有效性不存标志、用时现算: ``get_best(l, prefix)`` 天然 = "链式块最长存在到哪",
  目录里 "最深的 ≤ l 的位置" 即有效;条目被淘汰 => Get miss => 目录项自然作废,
  零通知、零跨层协议(4.3)。

本模块**零第三方依赖**(同 ``kv_spec_table``),可在未构建 C++ 扩展的裸环境下
直接单测;``SnapshotGroup`` 对目录只做鸭子类型(register/deepest_candidate/
on_get_miss),不 import 协调器层,保持 store 层不依赖集成层。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

__all__ = [
    "SnapshotEntry",
    "SnapshotStore",
    "SnapshotGroup",
]

# 无条目可用的哨兵(与协调器 NO_CHECKPOINT 语义一致)。
NO_BOUNDARY = 0


@dataclass
class SnapshotEntry:
    """一个位置键快照条目。"""

    payload: bytearray = field(default_factory=bytearray)
    # 热度(CLOCK-1bit 的近似,6.6.1/附录 E1 的演进方向);Touch 只做脉冲累加。
    heat: int = 0
    # 所有权: Put/Donate 写方移交后由 store 持有;Get 永不转移所有权(CoW)。
    owner: bool = True


class SnapshotStore:
    """位置键快照存储原型: 键 = (位置, 前缀哈希)。

    同一位置不同前缀、同一前缀不同位置都是不同条目。"位置"是检查点网格刻度
    (token 序号),"前缀哈希"隔离跨前缀内容(位置对、内容错不会命中,4.3)。

    ``root_dir`` 给定时启用**字节持久化**(9.1 阶段 2 "快照真正落盘"): Put/Donate
    落盘(原子写: 先写 ``.tmp`` 再 rename),Get 内存 miss 时按需读回,淘汰时删盘上
    文件。条目文件内容 = 裸字节(状态张量序列化由调用方负责,store 不碰模型语义)。
    """

    def __init__(
        self,
        group_name: str,
        *,
        max_entries: Optional[int] = None,
        root_dir: Optional[os.PathLike] = None,
    ) -> None:
        self.group_name = group_name
        self.max_entries = max_entries
        self._entries: dict[tuple[int, bytes], SnapshotEntry] = {}
        self.root_dir = os.fspath(root_dir) if root_dir is not None else None
        if self.root_dir is not None:
            os.makedirs(self.root_dir, exist_ok=True)

    # -- 磁盘路径(持久化布局) ------------------------------------------------

    @staticmethod
    def _key_file_name(position: int, prefix_hash: bytes) -> str:
        """条目文件名: 位置 + 前缀哈希(hex) 拼成单文件,位置变化即不同文件。

        两级目录分片(首 2 hex)避免单目录文件过多;与 Posix 内容寻址布局
        (分片目录 + 定长文件名)同构,见附录 E3。
        """
        ph = prefix_hash.hex()
        return os.path.join(ph[:2], f"{position:x}-{ph}")

    def _entry_path(self, position: int, prefix_hash: bytes) -> Optional[str]:
        if self.root_dir is None:
            return None
        return os.path.join(self.root_dir, self._key_file_name(position, prefix_hash))

    def _remove_file(self, position: int, prefix_hash: bytes) -> None:
        path = self._entry_path(position, prefix_hash)
        if path is None:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def _write_file(self, position: int, prefix_hash: bytes, payload: bytes) -> None:
        path = self._entry_path(position, prefix_hash)
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)  # 原子提交,读方永远看不到半截文件

    def _read_file(self, position: int, prefix_hash: bytes) -> Optional[bytes]:
        path = self._entry_path(position, prefix_hash)
        if path is None:
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple[int, bytes]) -> bool:
        return key in self._entries

    def positions(self, prefix_hash: bytes) -> list[int]:
        """该前缀在本组内已登记的位置(升序)。"""
        return sorted(p for p, ph in self._entries if ph == prefix_hash)

    def put(self, position: int, prefix_hash: bytes, payload: bytes) -> bool:
        """Put: 写入快照状态。

        首次提交获胜(7.4 去重语义: 同键重复 Put 视为已存在,不覆盖);写方缓冲
        会被复制进 store(所有权移交;需要零拷贝移交用 :meth:`donate`)。
        返回是否新建。
        """
        key = (position, prefix_hash)
        if key in self._entries:
            return False
        self._entries[key] = SnapshotEntry(payload=bytearray(payload), heat=1)
        self._write_file(position, prefix_hash, payload)
        self._evict_if_over_capacity()
        return True

    def get(self, position: int, prefix_hash: bytes) -> Optional[bytes]:
        """Get(=CoW): 命中则返回**副本**并累加热度;未命中返回 None。

        取用前复制防污染(5.1 / 7.6 R4): 调用方随意改返回值,不影响存储内条目。
        内存 miss 时按需读盘(惰性恢复): 盘上也没有才是真 miss。
        """
        entry = self._entries.get((position, prefix_hash))
        if entry is None:
            disk_payload = self._read_file(position, prefix_hash)
            if disk_payload is None:
                return None
            entry = SnapshotEntry(payload=bytearray(disk_payload), heat=0)
            self._entries[(position, prefix_hash)] = entry
        entry.heat += 1
        return bytes(entry.payload)

    def donate(self, position: int, prefix_hash: bytes, payload: bytearray) -> bool:
        """Donate: 零拷贝移交写方缓冲所有权(写方此后不得再改该缓冲)。

        首次提交获胜;移交后 store 直接持有同一对象,不做拷贝。
        """
        key = (position, prefix_hash)
        if key in self._entries:
            return False
        self._entries[key] = SnapshotEntry(payload=payload, heat=0, owner=True)
        self._write_file(position, prefix_hash, bytes(payload))
        self._evict_if_over_capacity()
        return True

    def touch(
        self, position: Optional[int] = None, prefix_hash: Optional[bytes] = None
    ) -> None:
        """Touch 脉冲(5.1 / 6.4): 会话热度翻译后的瞬时信号,只累加热度不落状态。

        参数为 None 表示不限该维度(全组脉冲)。
        """
        for key, entry in self._entries.items():
            pos, ph = key
            if position is not None and pos != position:
                continue
            if prefix_hash is not None and ph != prefix_hash:
                continue
            entry.heat += 1

    def heat_rank(self) -> list[tuple[int, bytes, int]]:
        """按热度升序返回 (位置, 前缀哈希, heat),供位置价值淘汰决策。"""
        return sorted(
            ((pos, ph, e.heat) for (pos, ph), e in self._entries.items()),
            key=lambda x: x[2],
        )

    def evict_lowest_heat(self, limit: int = 1) -> list[tuple[int, bytes]]:
        """位置价值淘汰(开放项,附录 E9 方向): 逐出热度最低条目。

        返回被逐出的键,调用方(协调器)据此对检查点目录做 ``on_get_miss``。
        持久化条目同时删除盘上文件(下次 Get miss -> 目录项自然作废,4.3)。
        """
        evicted: list[tuple[int, bytes]] = []
        for pos, ph, _heat in self.heat_rank()[:limit]:
            self._entries.pop((pos, ph), None)
            self._remove_file(pos, ph)
            evicted.append((pos, ph))
        return evicted

    def _evict_if_over_capacity(self) -> None:
        if self.max_entries is not None and len(self._entries) > self.max_entries:
            self.evict_lowest_heat(len(self._entries) - self.max_entries)

    def snapshot_files(self) -> list[str]:
        """盘上已落盘条目文件(相对 root_dir;无持久化时为空列表)。"""
        if self.root_dir is None or not os.path.isdir(self.root_dir):
            return []
        names: list[str] = []
        for shard in os.listdir(self.root_dir):
            shard_dir = os.path.join(self.root_dir, shard)
            if not os.path.isdir(shard_dir):
                continue
            for name in os.listdir(shard_dir):
                if not name.endswith(".tmp"):
                    names.append(f"{shard}/{name}")
        return sorted(names)


class SnapshotGroup:
    """快照组组合件: SnapshotStore + 检查点目录(惰性失效,4.3/9.1)。

    协调器经 ``get_best`` 取 "最深可用检查点" 处的快照:

    - 链式 l 收缩 => 目录 ``deepest_candidate(l)`` 自动够不着更深的条目(惰性失效);
    - 存储条目被淘汰(evict) => 下次 ``get_best`` miss => 目录项自然作废(调
      ``on_get_miss``),引擎退化为该段状态重推(漏命安全)。
    """

    def __init__(
        self,
        group_name: str,
        directory: object,
        *,
        store: Optional[SnapshotStore] = None,
        grid_alignment: int = 1,
    ) -> None:
        self.group_name = group_name
        self.directory = directory  # 鸭子类型: register / deepest_candidate / on_get_miss
        self.store = store or SnapshotStore(group_name)
        self.grid_alignment = grid_alignment

    def put_at_position(
        self, prefix_hash: bytes, position: int, payload: bytes
    ) -> bool:
        """写快照并登记目录(请求结束 / 定间隔 / 二次未见触发点由协调器决定)。"""
        created = self.store.put(position, prefix_hash, payload)
        if created:
            self.directory.register(position, prefix_hash)
        return created

    def get_best(self, l: int, prefix_hash: bytes) -> Optional[bytes]:
        """取 ≤ l 的最深可用检查点快照(CoW);无可用则 None 并作废对应目录项。"""
        boundary = self.directory.deepest_candidate(l, prefix_hash)
        if boundary == NO_BOUNDARY:
            return None
        payload = self.store.get(boundary, prefix_hash)
        if payload is None:
            # 条目被淘汰/已收缩: 该目录项自然作废(4.3 get-miss 路径)。
            self.directory.on_get_miss(boundary, prefix_hash)
            return None
        return payload

    def evict(self, limit: int = 1) -> list[tuple[int, bytes]]:
        """逐出存储条目并同步作废目录项,返回被逐出的键。"""
        evicted = self.store.evict_lowest_heat(limit)
        for pos, ph in evicted:
            self.directory.on_get_miss(pos, ph)
        return evicted


# 供协调器复用的类型别名(6.6.2 边界: 协调器懂模型、不碰字节)。
BoundarySelector = Callable[[int, bytes], Optional[bytes]]
