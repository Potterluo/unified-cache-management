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

"""Unit tests for the coordinator stage-1 primitives (spec table / resolve_hit /
checkpoint directory), with assertions taken from 4.5 算例 A/B/C of the design
report (UCM_缓存设计分析报告_2026-08.md).

``kv_spec_table`` is a zero-dependency module (no vllm / torch / C++ extension),
so it is loaded here directly from its source file instead of importing the
``ucm`` package (which requires the built extension for ``ucm.logger``).
"""

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "ucm" / "integration" / "vllm" / "kv_spec_table.py"


def _load_spec_table_module():
    spec = importlib.util.spec_from_file_location("kv_spec_table", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["kv_spec_table"] = module
    spec.loader.exec_module(module)
    return module


# Executed once at import time; the module itself only imports the stdlib.
kv_spec_table = _load_spec_table_module()

CacheKind = kv_spec_table.CacheKind
RankRule = kv_spec_table.RankRule
RetentionPolicy = kv_spec_table.RetentionPolicy
SpecRow = kv_spec_table.SpecRow
SpecTable = kv_spec_table.SpecTable
CheckpointDirectory = kv_spec_table.CheckpointDirectory
resolve_hit = kv_spec_table.resolve_hit
rank_block_present = kv_spec_table.rank_block_present
deepest_snapshot_p_star = kv_spec_table.deepest_snapshot_p_star
NO_CHECKPOINT = kv_spec_table.NO_CHECKPOINT


def _xhybrid_96_spec_table():
    """4.1 规格表示例(仅取 4.5 算例用到的组;字节数为示例值)。"""
    return SpecTable(
        [
            SpecRow(
                "mla", CacheKind.CHAIN, 128, (512,) * 16, "S_mla", RankRule.ALL_UNION
            ),
            SpecRow(
                "csa_c16",
                CacheKind.CHAIN,
                128,
                (64,) * 16,
                "S_c16",
                RankRule.ALL_UNION,
            ),
            SpecRow(
                "swa", CacheKind.CHAIN, 128, (128,) * 24, "S_swa", RankRule.ALL_UNION
            ),
            SpecRow(
                "mamba2",
                CacheKind.SNAPSHOT,
                64,
                (512,) * 20,
                "S_m2",
                RankRule.ALL_UNION,
                RetentionPolicy(interval=1024, grid_alignment=64),
            ),
        ]
    )


class SpecTableTest(unittest.TestCase):
    def test_lcm_block_size_from_chain_groups(self):
        spec = _xhybrid_96_spec_table()
        self.assertEqual(spec.lcm_block_size, 128)
        self.assertEqual(
            [r.group_name for r in spec.chain_rows], ["mla", "csa_c16", "swa"]
        )
        self.assertEqual([r.group_name for r in spec.snapshot_rows], ["mamba2"])

    def test_duplicate_group_name_rejected(self):
        rows = [
            SpecRow("mla", CacheKind.CHAIN, 128),
            SpecRow("mla", CacheKind.CHAIN, 64),
        ]
        with self.assertRaises(ValueError):
            SpecTable(rows)

    def test_none_kind_requires_no_block_size(self):
        with self.assertRaises(AssertionError):
            SpecRow("cross", CacheKind.NONE, 128)
        row = SpecRow("cross", CacheKind.NONE, None)
        self.assertIsNone(row.block_size)


class ResolveHitTest(unittest.TestCase):
    """4.2 组件投票 + 4.5 算例 A 的单测断言。"""

    def _lookup_ctx(self, existence_by_group):
        """构造 (spec, prefix_hashes, chain_existence) 最小上下文。"""

        def chain_existence(row, block_ids):
            return existence_by_group[row.group_name]

        return {
            "chain_existence": chain_existence,
            "prefix_hashes": {
                g: [bytes([i]) for i in range(64)] for g in existence_by_group
            },
        }

    def test_example_A_component_voting(self):
        # 5000 token 前缀: MLA 查到 4608(块 36),csa_c16 查到 4480(块 35),
        # swa 查到 5000(窗口内);检查点目录 {4096, 4608}。
        spec = _xhybrid_96_spec_table()
        ctx = self._lookup_ctx({"mla": 4608, "csa_c16": 4480, "swa": 5000})
        mamba2 = CheckpointDirectory("mamba2", grid_alignment=64)
        mamba2.register(4096, b"prefix-a")
        mamba2.register(4608, b"prefix-a")

        l, p_star = resolve_hit(
            spec,
            ctx["prefix_hashes"],
            ctx["chain_existence"],
            {"mamba2": mamba2},
            {"mamba2": b"prefix-a"},
        )
        # LCM = lcm(128,128,128) = 128;min(4608,4480,5000) = 4480;4480 % 128 == 0。
        # p*: 4608 > l 够不着,4096 <= l -> p* = 4096。
        self.assertEqual((l, p_star), (4480, 4096))

    def test_example_A_min_protection_against_miss_hit(self):
        # 错命演示: 只看 MLA 的 4608 就跳过,会踩到 csa 不存在的 [4480,4608);
        # 组件投票取交集最多重算 -> 安全。
        spec = _xhybrid_96_spec_table()
        ctx = self._lookup_ctx({"mla": 4608, "csa_c16": 4480, "swa": 5000})
        mamba2 = CheckpointDirectory("mamba2", grid_alignment=64)
        mamba2.register(4096, b"prefix-a")

        l, _ = resolve_hit(
            spec,
            ctx["prefix_hashes"],
            ctx["chain_existence"],
            {"mamba2": mamba2},
            {"mamba2": b"prefix-a"},
        )
        self.assertEqual(l, 4480)
        self.assertLess(l, 4608)

    def test_floor_to_lcm_when_groups_have_different_block_sizes(self):
        spec = SpecTable(
            [
                SpecRow("mla", CacheKind.CHAIN, 128),
                SpecRow("csa_d64", CacheKind.CHAIN, 64),
            ]
        )
        self.assertEqual(spec.lcm_block_size, 128)
        # min(4608, 4416) = 4416,向下对齐 LCM=128 -> 4352(部分块不算命中)。
        ctx = self._lookup_ctx({"mla": 4608, "csa_d64": 4416})
        l, p_star = resolve_hit(
            spec,
            ctx["prefix_hashes"],
            ctx["chain_existence"],
            {},
        )
        self.assertEqual((l, p_star), (4352, 4352))

    def test_no_snapshot_groups_returns_p_star_equals_l(self):
        spec = SpecTable([SpecRow("mla", CacheKind.CHAIN, 128)])
        ctx = self._lookup_ctx({"mla": 4096})
        l, p_star = resolve_hit(spec, ctx["prefix_hashes"], ctx["chain_existence"], {})
        self.assertEqual((l, p_star), (4096, 4096))

    def test_alignment_override_canonical_grid_preserves_hits(self):
        # FAWA 场景(4.2 对齐的适配): 链式组 token block 的 LCM 很大(如 DSV4
        # mla1 组 4096),但 FAWA 存储层以 canonical hash 块(128)为复用单位,
        # 命中必须保留 128 刻度 -- 用 alignment 覆盖 LCM,防止 4096 对齐把
        # 3712 的命中截到 0。
        spec = SpecTable(
            [
                SpecRow("mla0", CacheKind.CHAIN, 128),
                SpecRow("mla1", CacheKind.CHAIN, 4096),
            ]
        )
        self.assertEqual(spec.lcm_block_size, 4096)
        # 两 FA 组共享同一 canonical 前缀: min(3712, 3712)。
        ctx = self._lookup_ctx({"mla0": 3712, "mla1": 3712})

        l_default, _ = resolve_hit(
            spec, ctx["prefix_hashes"], ctx["chain_existence"], {}
        )
        # 缺省按规格表 LCM=4096 对齐: 3712 -> 0(命中被错误截断)。
        self.assertEqual(l_default, 0)

        l_grid, _ = resolve_hit(
            spec,
            ctx["prefix_hashes"],
            ctx["chain_existence"],
            {},
            alignment=128,
        )
        # canonical 刻度 128 对齐: 3712 % 128 == 0 -> 保留。
        self.assertEqual(l_grid, 3712)

    def test_chain_row_filter_excludes_window_groups(self):
        # FAWA 的窗口组(Tail/SWA)不是前缀数据,不参与链式投票(经
        # lookup_on_reverse 单独裁决,见 hma_connector);row_filter 使其
        # 从 resolve_hit 的 chain 投票中排除,避免窗口组把 l 拖成 0。
        spec = SpecTable(
            [
                SpecRow("fa0", CacheKind.CHAIN, 128),
                SpecRow("wa0", CacheKind.CHAIN, 32),
            ]
        )
        ctx = self._lookup_ctx({"fa0": 3712, "wa0": 0})

        l_default, _ = resolve_hit(
            spec, ctx["prefix_hashes"], ctx["chain_existence"], {}
        )
        # 缺省窗口组参与投票: min(3712, 0) = 0,命中被窗口组抹掉。
        self.assertEqual(l_default, 0)

        l_filtered, _ = resolve_hit(
            spec,
            ctx["prefix_hashes"],
            ctx["chain_existence"],
            {},
            chain_row_filter=lambda row: row.group_name == "fa0",
        )
        self.assertEqual(l_filtered, 3712)

    def test_snapshot_group_without_prefix_id_contributes_zero(self):
        spec = _xhybrid_96_spec_table()
        ctx = self._lookup_ctx({"mla": 4608, "csa_c16": 4480, "swa": 5000})
        mamba2 = CheckpointDirectory("mamba2")
        mamba2.register(4096, b"other-prefix")
        l, p_star = resolve_hit(
            spec,
            ctx["prefix_hashes"],
            ctx["chain_existence"],
            {"mamba2": mamba2},
            {"mamba2": b"prefix-a"},  # 目录里只有 other-prefix 的条目
        )
        self.assertEqual(l, 4480)
        self.assertEqual(p_star, NO_CHECKPOINT)


class RankRuleTest(unittest.TestCase):
    """4.5 算例 B: 秩规则是"正确性 × 冗余策略"。"""

    def test_intersection_requires_all_ranks(self):
        self.assertFalse(
            rank_block_present(RankRule.ALL_INTERSECT, [True, True, True, False])
        )
        self.assertTrue(rank_block_present(RankRule.ALL_INTERSECT, [True] * 4))

    def test_union_requires_any_rank(self):
        self.assertTrue(
            rank_block_present(RankRule.ALL_UNION, [False, False, False, True])
        )
        self.assertFalse(rank_block_present(RankRule.ALL_UNION, [False] * 4))

    def test_example_B_availability(self):
        # 非 MLA(4 rank,p=0.1,独立): 可用概率 (1-p)^4 ≈ 65.6%(交集);
        # MLA: 任一 rank dump 成功即可: 1-p^4 ≈ 99.99%(并集)。
        rng = random.Random(20260828)
        n = 20000
        p = 0.1
        intersect_hits = union_hits = 0
        for _ in range(n):
            success = [rng.random() >= p for _ in range(4)]
            if rank_block_present(RankRule.ALL_INTERSECT, success):
                intersect_hits += 1
            if rank_block_present(RankRule.ALL_UNION, success):
                union_hits += 1
        self.assertAlmostEqual(intersect_hits / n, 0.6561, delta=0.02)
        self.assertAlmostEqual(union_hits / n, 0.9999, delta=0.0005)


class CheckpointDirectoryTest(unittest.TestCase):
    """4.3 检查点目录 + 惰性失效 + 4.5 算例 C 的目录增长断言。"""

    def _prefix_dir(self):
        return CheckpointDirectory("mamba2", grid_alignment=1)

    def test_example_C_dir_growth(self):
        # 10000 token 长输入,间隔 1024。
        d = self._prefix_dir()
        prefix = b"same-prefix"

        # 请求1 完整跑完: {1024, 2048, ..., 9216, 10000}(定间隔触发)。
        created = d.on_request_ended(prefix, 10000, interval=1024)
        expected1 = set(range(1024, 10001, 1024)) | {10000}
        self.assertEqual(sorted(created), sorted(expected1 - {0}))
        self.assertEqual(d.positions(prefix), expected1)

        # 请求2 相同前缀,全部命中(到 10000): 无新增内容,跳过 -> 目录不变。
        self.assertEqual(d.deepest_candidate(10000, prefix), 10000)
        self.assertEqual(d.positions(prefix), expected1)

        # 请求3 前缀相同但只到 5000: 链式块命中 5000,检查点 @5000 缺失
        # -> 从最深检查点 4096 续算;请求结束新增 {5000}。
        self.assertEqual(d.deepest_candidate(5000, prefix), 4096)
        new_created = d.on_request_ended(prefix, 5000, interval=1024)
        self.assertEqual(new_created, [5000])
        self.assertEqual(d.positions(prefix), expected1 | {5000})

        # 请求4 同前缀: 边界已存在 -> 不重复创建。
        again = d.on_request_ended(prefix, 5000, interval=1024)
        self.assertEqual(again, [])
        self.assertEqual(d.positions(prefix), expected1 | {5000})

    def test_lazy_invalidation_blocks_evicted(self):
        # 块被淘汰 => 检查点自动够不着: 不存有效标志、零通知、零跨层协议。
        d = self._prefix_dir()
        prefix = b"p"
        d.register(4096, prefix)
        d.register(4608, prefix)
        # 链式块最长存在到 4480: 4096 有效。
        self.assertEqual(d.deepest_candidate(4480, prefix), 4096)
        # 链式块被淘汰到只剩 3072: 两个检查点都够不着 -> NO_CHECKPOINT,状态重推。
        self.assertEqual(d.deepest_candidate(3072, prefix), NO_CHECKPOINT)
        self.assertEqual(d.deepest_candidate(0, prefix), NO_CHECKPOINT)

    def test_get_miss_invalidates_entry(self):
        d = self._prefix_dir()
        prefix = b"p"
        d.register(4096, prefix)
        d.register(4608, prefix)
        d.on_get_miss(4096, prefix)  # 快照条目被存储淘汰
        self.assertEqual(d.deepest_candidate(5000, prefix), 4608)
        d.on_get_miss(4608, prefix)
        self.assertEqual(d.deepest_candidate(5000, prefix), NO_CHECKPOINT)

    def test_key_includes_prefix_hash_cross_prefix_isolation(self):
        # 目录的键必须含前缀哈希: "位置对、内容错" 的跨前缀错命不会发生。
        d = self._prefix_dir()
        d.register(4096, b"prefix-A")
        self.assertEqual(d.deepest_candidate(4096, b"prefix-A"), 4096)
        self.assertEqual(d.deepest_candidate(4096, b"prefix-B"), NO_CHECKPOINT)

    def test_grid_alignment(self):
        d = CheckpointDirectory("mamba2", grid_alignment=64)
        d.register(4096, b"p")
        d.register(4064, b"p")  # 向下对齐到 64 的倍数(4064 % 64 == 32 -> 4032)
        self.assertEqual(sorted(d.positions(b"p")), [4032, 4096])
        d.register(4064, b"p")  # 重复登记幂等
        self.assertEqual(sorted(d.positions(b"p")), [4032, 4096])

    def test_second_unseen_trigger(self):
        # 保留策略触发②: 同一前缀第二次"未被服务"-> 公共边界存;命中过的不算。
        d = self._prefix_dir()
        prefix = b"p"
        self.assertFalse(d.on_unserved_seen(prefix, 5000))
        self.assertEqual(d.deepest_candidate(5000, prefix), NO_CHECKPOINT)
        self.assertTrue(d.on_unserved_seen(prefix, 5000))
        self.assertEqual(d.deepest_candidate(5000, prefix), 5000)
        # 第三次出现不再新建。
        self.assertFalse(d.on_unserved_seen(prefix, 5000))


class DeepestSnapshotPStarTest(unittest.TestCase):
    """4.3 目录驱动 p* 纯函数: 最深 ≤ l 且所有快照组齐备的位置。

    检查点前缀哈希随位置链式变化(位置隔离): 位置 p 只对该位置自己的前缀哈希
    可见;所有快照组在同一 p 都登记过才推进。
    """

    def _prefix_at(self, hashes_by_pos):
        def prefix_at(position):
            return hashes_by_pos.get(position)
        return prefix_at

    def test_deepest_across_all_snapshot_groups(self):
        # 两组快照: 在 4096 都登记 -> p* = 4096;4608 只有一组 -> 不可用。
        m2 = CheckpointDirectory("m2")
        kda = CheckpointDirectory("kda")
        prefix_4096, prefix_4608 = b"h4096", b"h4608"
        m2.register(4096, prefix_4096)
        m2.register(4608, prefix_4608)
        kda.register(4096, prefix_4096)  # kda 在 4608 缺失
        p_star = deepest_snapshot_p_star(
            {"m2": m2, "kda": kda},
            self._prefix_at({4096: prefix_4096, 4608: prefix_4608}),
            num_computed_tokens=0,
            total_hit_tokens=4608,
            lcm_block_size=128,
        )
        self.assertEqual(p_star, 4096)

    def test_cross_prefix_isolation_by_position_hash(self):
        # 同一位置不同前缀哈希: 目录项对不上 -> 不可用(位置对、内容错不命中)。
        d = CheckpointDirectory("m2")
        d.register(4096, b"prefix-A-at-4096")
        p_star = deepest_snapshot_p_star(
            {"m2": d},
            self._prefix_at({4096: b"prefix-B-at-4096"}),
            num_computed_tokens=0,
            total_hit_tokens=4096,
            lcm_block_size=128,
        )
        self.assertEqual(p_star, 0)  # NO_CHECKPOINT -> 状态重推

    def test_no_checkpoint_below_l_returns_computed(self):
        d = CheckpointDirectory("m2")
        d.register(1024, b"h1024")
        # 链式候选 4096,但最深已登记检查点只有 1024 -> p* = 1024。
        p_star = deepest_snapshot_p_star(
            {"m2": d},
            self._prefix_at({1024: b"h1024", 4096: b"h4096"}),
            num_computed_tokens=0,
            total_hit_tokens=4096,
            lcm_block_size=128,
        )
        self.assertEqual(p_star, 1024)

    def test_empty_directories_returns_total_hit(self):
        p_star = deepest_snapshot_p_star(
            {},
            self._prefix_at({}),
            num_computed_tokens=0,
            total_hit_tokens=4096,
            lcm_block_size=128,
        )
        self.assertEqual(p_star, 4096)

    def test_partial_prefix_hash_skips_position(self):
        # 前缀哈希在某个位置缺失(链式块被淘汰) -> 该位置跳过(惰性失效)。
        m2 = CheckpointDirectory("m2")
        m2.register(4096, b"h4096")
        m2.register(4608, b"h4608")
        # prefix_at(4608) 返回 None => 4608 不可用,退化到 4096。
        p_star = deepest_snapshot_p_star(
            {"m2": m2},
            self._prefix_at({4096: b"h4096"}),
            num_computed_tokens=0,
            total_hit_tokens=4608,
            lcm_block_size=128,
        )
        self.assertEqual(p_star, 4096)


if __name__ == "__main__":
    unittest.main()
