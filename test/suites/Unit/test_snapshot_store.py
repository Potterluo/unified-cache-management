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

"""Stage-2 unit tests: SnapshotStore (5.1 位置键 / CoW / Donate / Touch) +
SnapshotGroup 惰性失效(4.3),断言口径与 9.1 阶段 2 一致。

与 test_kv_spec_table 相同的零依赖加载方式(importlib 直接加载模块文件)。
"""

import math
import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_PATH = REPO_ROOT / "ucm" / "store" / "snapshot_store.py"
KVS_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "kv_spec_table.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


snapshot_store = _load_module("snapshot_store", SNAPSHOT_PATH)
kv_spec_table = _load_module("kv_spec_table", KVS_PATH)

SnapshotStore = snapshot_store.SnapshotStore
SnapshotGroup = snapshot_store.SnapshotGroup
CheckpointDirectory = kv_spec_table.CheckpointDirectory
NO_BOUNDARY = snapshot_store.NO_BOUNDARY


class SnapshotStoreTest(unittest.TestCase):
    """5.1: 位置键 / CoW / Donate / Touch / 位置价值淘汰。"""

    def test_position_key_semantics(self):
        # "罐头": 位置不对就废;同一位置不同前缀互不可见。
        s = SnapshotStore("mamba2")
        self.assertTrue(s.put(4096, b"prefix-A", b"state-a"))
        self.assertEqual(s.get(4096, b"prefix-A"), b"state-a")
        self.assertIsNone(s.get(4096, b"prefix-B"))  # 前缀不对
        self.assertIsNone(s.get(4032, b"prefix-A"))  # 位置不对

    def test_put_first_commit_wins(self):
        s = SnapshotStore("mamba2")
        self.assertTrue(s.put(4096, b"p", b"first"))
        self.assertFalse(s.put(4096, b"p", b"second"))
        self.assertEqual(s.get(4096, b"p"), b"first")

    def test_get_is_cow(self):
        s = SnapshotStore("mamba2")
        s.put(4096, b"p", b"hello-kv")
        got = s.get(4096, b"p")
        assert got is not None
        mutated = bytearray(got)
        mutated[0] = ord("X")
        self.assertEqual(s.get(4096, b"p"), b"hello-kv")  # 存储不受影响
        self.assertNotEqual(bytes(mutated), s.get(4096, b"p"))

    def test_donate_zero_copy_takeover(self):
        s = SnapshotStore("mamba2")
        buf = bytearray(b"donated-state")
        self.assertTrue(s.donate(2048, b"p", buf))
        # 零拷贝: store 直接持有同一缓冲对象(不做拷贝)。
        entry = s._entries[(2048, b"p")]
        self.assertIs(entry.payload, buf)
        # Get 仍返回副本,防调用方在共享态上改写。
        got = s.get(2048, b"p")
        self.assertIsNot(got, buf)
        self.assertEqual(got, b"donated-state")

    def test_touch_and_evict_lowest_heat(self):
        s = SnapshotStore("mamba2")
        s.put(1024, b"p", b"a")
        s.put(2048, b"p", b"b")
        s.put(3072, b"p", b"c")
        s.touch()  # 全组脉冲
        s.touch(prefix_hash=b"p")
        s.get(2048, b"p")  # 2048 再热一点
        evicted = s.evict_lowest_heat(1)
        self.assertEqual(evicted, [(1024, b"p")])
        self.assertIsNone(s.get(1024, b"p"))
        self.assertIsNotNone(s.get(3072, b"p"))

    def test_max_entries_capacity_eviction(self):
        s = SnapshotStore("kda", max_entries=2)
        s.put(1, b"p", b"x1")
        s.put(2, b"p", b"x2")
        s.put(3, b"p", b"x3")  # 超出容量 -> 逐出热度最低
        self.assertEqual(len(s), 2)
        self.assertIsNotNone(s.get(3, b"p"))

    def test_disk_persistence_put_get_across_instances(self):
        # 9.1 阶段 2 字节落盘: Put 落盘 -> 新实例从磁盘惰性读回 -> CoW 语义不变。
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            s1 = SnapshotStore("mamba2", root_dir=tmp)
            self.assertTrue(s1.put(3072, b"prefix", b"\x00\x01state-bytes"))
            self.assertTrue(s1.put(4096, b"prefix", b"\x02\x03other"))
            # 原子写: 无 .tmp 残留,两个条目各一个文件。
            files = s1.snapshot_files()
            self.assertEqual(len(files), 2)
            self.assertFalse(any(f.endswith(".tmp") for f in files))

            # 新实例(空内存)按需读盘。
            s2 = SnapshotStore("mamba2", root_dir=tmp)
            self.assertEqual(len(s2), 0)
            self.assertEqual(s2.get(3072, b"prefix"), b"\x00\x01state-bytes")
            self.assertEqual(s2.get(4096, b"prefix"), b"\x02\x03other")
            self.assertEqual(len(s2), 2)

            # CoW: 修改返回值不影响存储内条目(读回后仍有原值)。
            got = s2.get(3072, b"prefix")
            self.assertIsNotNone(got)
            mutated = bytearray(got)
            mutated[0] = 0xFF
            self.assertEqual(s2.get(3072, b"prefix"), b"\x00\x01state-bytes")

            # 位置键语义持久化不变: 位置差一个 token 即不同条目。
            self.assertIsNone(s2.get(3073, b"prefix"))
            self.assertIsNone(s2.get(3072, b"other-prefix"))

    def test_disk_eviction_removes_file_and_auto_invalidates(self):
        # 淘汰 = 内存条目 + 盘上文件一起删;新实例读回 miss -> 目录项作废(4.3)。
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            s = SnapshotStore("mamba2", root_dir=tmp)
            s.put(1024, b"p", b"a")
            s.put(2048, b"p", b"b")
            s.touch()
            s.get(2048, b"p")  # 2048 更热

            evicted = s.evict_lowest_heat(1)
            self.assertEqual(evicted, [(1024, b"p")])
            self.assertEqual(len(s.snapshot_files()), 1)  # 盘上同步删除

            s2 = SnapshotStore("mamba2", root_dir=tmp)
            self.assertIsNone(s2.get(1024, b"p"))  # 读盘 miss
            self.assertEqual(s2.get(2048, b"p"), b"b")  # 未淘汰条目可读回


class PositionValueEvictionTest(unittest.TestCase):
    """5.2 Gap 初版: 快照位置价值 = 频率 × 时间衰减。"""

    def test_score_monotonic_heat_and_age(self):
        score = snapshot_store.position_value_score
        # 热且新 > 冷且旧
        self.assertGreater(score(10, age=1), score(1, age=1e6))
        # 同 heat 下 age 小者分高
        self.assertGreater(score(5, age=10), score(5, age=1000))
        # 同 age 下 heat 高者分高
        self.assertGreater(score(7, age=100), score(2, age=100))
        # 半衰期: 一个半衰期后价值 ~一半(时间维衰减)
        self.assertAlmostEqual(
            score(1, 0) / score(1, 3600), math.e, delta=0.01
        )

    def test_evict_lowest_value_picks_coldest_coldest(self):
        store = snapshot_store.SnapshotStore("g1")
        now = snapshot_store._monotonic()
        store.put(100, b"a" * 8, b"A")  # heat=1, t=now
        store.put(200, b"a" * 8, b"B")  # heat=1, t=now
        # 让 (200,*) 变冷: 手动改 last_touch
        store._entries[(200, b"a" * 8)].last_touch = now - 100000
        evicted = store.evict_lowest_value(limit=1, now=now)
        self.assertEqual(evicted, [(200, b"a" * 8)])
        self.assertIn((100, b"a" * 8), store._entries)

    def test_evict_lowest_value_frequency_overrides_age(self):
        store = snapshot_store.SnapshotStore("g1")
        now = snapshot_store._monotonic()
        store.put(100, b"x" * 8, b"A")
        store.put(200, b"x" * 8, b"B")
        # 位置100 频率高(heat=9)但同样冷;位置200 频率低 -> 逐出 200
        store._entries[(100, b"x" * 8)].heat = 9
        store._entries[(200, b"x" * 8)].heat = 1
        for key in store._entries:
            store._entries[key].last_touch = now - 100000
        evicted = store.evict_lowest_value(limit=1, now=now)
        self.assertEqual(evicted, [(200, b"x" * 8)])


class SharedSnapshotStoreTest(unittest.TestCase):
    """7.6 R4: 进程内共享快照存储注册表(dump Put / load Get 同实例)。"""

    def tearDown(self):
        snapshot_store._snapshot_store_registry.clear()

    def test_same_group_key_shares_instance(self):
        a = snapshot_store.shared_snapshot_store((3072, 1), "g1")
        b = snapshot_store.shared_snapshot_store((3072, 1), "g1")
        self.assertIs(a, b)

    def test_different_group_key_distinct(self):
        a = snapshot_store.shared_snapshot_store((3072, 1), "g1")
        b = snapshot_store.shared_snapshot_store((3072, 2), "g2")
        c = snapshot_store.shared_snapshot_store((1024, 1), "g3")
        self.assertIsNot(a, b)
        self.assertIsNot(a, c)
        self.assertIsNot(b, c)

    def test_put_get_across_instances(self):
        # worker 侧与 SCHEDULER 侧各自调用 shared_snapshot_store,必须互见。
        worker = snapshot_store.shared_snapshot_store((3072, 1), "g1")
        scheduler = snapshot_store.shared_snapshot_store((3072, 1), "g1")
        prefix = b"p" * 16
        self.assertTrue(worker.put(3072, prefix, b"state-bytes"))
        # 重复 Put 幂等(首次提交获胜,7.4)。
        self.assertFalse(worker.put(3072, prefix, b"other"))
        self.assertEqual(scheduler.get(3072, prefix), b"state-bytes")
        # Get = CoW: 改返回值不影响存储内条目。
        got = scheduler.get(3072, prefix)
        got += b"x"
        self.assertEqual(worker.get(3072, prefix), b"state-bytes")

    def test_registry_survives_reload_calls(self):
        # 注册表跨连接器实例化(worker/scheduler 各 new 一次 connector)保持。
        first = snapshot_store.shared_snapshot_store((3072, 3), "g3")
        first.put(3072, b"k" * 8, b"v")
        second = snapshot_store.shared_snapshot_store((3072, 3), "g3")
        self.assertIs(first, second)
        self.assertEqual(second.get(3072, b"k" * 8), b"v")


class SnapshotGroupTest(unittest.TestCase):
    """4.3 惰性失效 + 算例 C 风格的目录联动。"""

    def _group(self, grid=64):
        directory = CheckpointDirectory("mamba2", grid_alignment=grid)
        return SnapshotGroup("mamba2", directory, grid_alignment=grid), directory

    def test_get_best_deepest_within_l(self):
        group, _dir = self._group()
        group.put_at_position(b"p", 4096, b"state-4096")
        group.put_at_position(b"p", 4608, b"state-4608")
        # l=4480: 4608 够不着,取最深的 4096。
        self.assertEqual(group.get_best(4480, b"p"), b"state-4096")
        self.assertEqual(group.get_best(5000, b"p"), b"state-4608")

    def test_lazy_invalidation_when_chain_shrinks(self):
        group, directory = self._group()
        group.put_at_position(b"p", 4096, b"state")
        self.assertEqual(group.get_best(4096, b"p"), b"state")
        # 链式块被淘汰到只剩 3072: 检查点自动够不着 -> None(零通知零跨层)。
        self.assertIsNone(group.get_best(3072, b"p"))
        self.assertEqual(directory.deepest_candidate(3072, b"p"), NO_BOUNDARY)

    def test_evicted_entry_auto_invalidates_directory(self):
        group, directory = self._group()
        group.put_at_position(b"p", 4096, b"state")
        self.assertEqual(directory.positions(b"p"), {4096})
        group.evict(limit=1)
        # 目录项已作废;深位置也取不到(条目没了)。
        self.assertEqual(directory.positions(b"p"), set())
        self.assertIsNone(group.get_best(9999, b"p"))

    def test_cross_prefix_isolation(self):
        group, directory = self._group()
        group.put_at_position(b"A", 4096, b"state-A")
        group.put_at_position(b"B", 4096, b"state-B")
        self.assertEqual(group.get_best(4096, b"A"), b"state-A")
        self.assertEqual(group.get_best(4096, b"B"), b"state-B")
        self.assertEqual(directory.positions(b"A"), {4096})

    def test_example_C_style_request_end_growth(self):
        # 10000 token 输入,定间隔 1024: 请求1 完整跑完登记到 10000;
        # 请求3 只到 5000 -> 从最深检查点 4096 续算(p* 语义),请求结束补 {5000}。
        group, directory = self._group(grid=1)
        prefix = b"same-prefix"
        end1 = 10000
        for pos in list(range(1024, end1 + 1, 1024)) + [end1]:
            group.put_at_position(prefix, pos, b"state@%d" % pos)
        self.assertEqual(group.get_best(end1, prefix), b"state@10000")
        # 请求3: 链式命中到 5000,但检查点 @5000 缺失 -> p* = 4096。
        self.assertEqual(group.get_best(5000, prefix), b"state@4096")
        group.put_at_position(prefix, 5000, b"state@5000")
        self.assertIn(5000, directory.positions(prefix))


if __name__ == "__main__":
    unittest.main()
