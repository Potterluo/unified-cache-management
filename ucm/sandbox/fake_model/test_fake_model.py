"""假模型工具集的自包含单元测试(纯 stdlib,不依赖 vllm/torch/ucm 包)。

按任务要求可直接这样跑(从仓库根目录):
    python -m pytest ucm/sandbox/fake_model/test_fake_model.py \
        -c test/pytest.ini --confcutdir=test/suites/Unit
--confcutdir 保证不会加载 test/conftest.py(它会拖入 pynvml 等依赖);
本文件不 import 任何 ucm 包(ucm/__init__.py 会拖入 vllm patch),只通过
sys.path 加载同目录的三个工具模块。

覆盖点:
  (a) 缩减 config 的层数 == N;
  (b) 前 N 层层类型模式与原始/模板逐层一致;
  (c) KV 关键形状字段在缩减后原样保留;
  (d) layer_plan.json 结构合法(必填键/逐层序号/组合法性/来源标注);
  (e) 报告口径模板可独立工作(离线兜底);
  (f) 研究模式(无 --out)只打印合法 JSON;
  (g) 随机 safetensors(纯 stdlib)可写可读回。
"""

import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # 自包含:不依赖 ucm 包/仓库根目录
    sys.path.insert(0, str(_HERE))

import build_fake_model  # noqa: E402  本目录模块,仅 stdlib 依赖
import fake_weights  # noqa: E402
import layer_plans as lp  # noqa: E402

MODELS = ("deepseek-v4", "kimi-k3", "glm-5.3")
OFFICIAL_DIR = _HERE / "official_configs"

# 报告 6.1 / 官方 config 给出的"官方口径"前 8 层类型串
EXPECTED_FIRST8 = {
    "deepseek-v4": "full,full,c4,c128,c4,c128,c4,c128",  # FULL×2 + CSA C4/C128 交替
    "kimi-k3": "mla,kda,kda,kda,mla,kda,kda,kda",  # MLА 每 4 层 1 个,首层 dense
    "glm-5.3": "kda,kda,kda,dsa,kda,kda,kda,dsa",  # 34 KDA 线性 + 11 DeepSeek 稀疏
}
EXPECTED_TOTAL = {
    "deepseek-v4": 43,  # 2 full + 38 csa + 3 dspark(target 40~42)
    "kimi-k3": 93,  # 24 MLA + 69 KDA(MLA 每 4 层 + 首层 dense)
    "glm-5.3": 45,  # 34 KDA + 11 DSA
}


def _load_official(model_key: str) -> dict:
    path = OFFICIAL_DIR / (model_key + ".json")
    if not path.exists():
        raise unittest.SkipTest(f"缺少官方缓存 config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_for(model_key: str, cfg: dict, n: int) -> dict:
    """对给定 config 建 plan(与 build_fake_model 同一条路径)。"""
    source = lp.template_source(model_key)
    return lp.build_layer_plan(model_key, cfg, n, source)


def _run_cli(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = build_fake_model.main(argv)
    return rc, out.getvalue()


class TestLayerClassification(unittest.TestCase):
    """(b) 核心:官方 config 逐层分类的正确性 + 模式一致性。"""

    def test_official_layer_type_patterns(self):
        for model_key in MODELS:
            with self.subTest(model=model_key):
                cfg = _load_official(model_key)
                full = lp.classify_layers(model_key, cfg)
                self.assertEqual(
                    len(full),
                    EXPECTED_TOTAL[model_key],
                    f"{model_key}: 官方层数与报告口径不一致",
                )
                type_str = ",".join(lp.compact_type(e) for e in full[:8])
                self.assertEqual(type_str, EXPECTED_FIRST8[model_key])

    def test_official_total_kind_counts(self):
        cfg = _load_official("deepseek-v4")
        kinds = [e["type"] for e in lp.classify_layers("deepseek-v4", cfg)]
        self.assertEqual(kinds.count("full"), 2)
        self.assertEqual(kinds.count("csa"), 38)
        self.assertEqual(kinds.count("dspark"), 3)

        cfg = _load_official("kimi-k3")
        kinds = [e["type"] for e in lp.classify_layers("kimi-k3", cfg)]
        self.assertEqual(kinds.count("mla"), 24)
        self.assertEqual(kinds.count("kda"), 69)

        cfg = _load_official("glm-5.3")
        kinds = [e["type"] for e in lp.classify_layers("glm-5.3", cfg)]
        self.assertEqual(kinds.count("dsa"), 11)
        self.assertEqual(kinds.count("kda"), 34)
        mlps = [e["mlp"] for e in lp.classify_layers("glm-5.3", cfg)]
        self.assertEqual(mlps[:3], ["dense"] * 3)
        self.assertTrue(all(m == "moe" for m in mlps[3:]))

    def test_reduced_pattern_matches_original_prefix(self):
        """(b) 缩减后前 N 层的类型模式 == 原始前 N 层(逐层比)。"""
        for model_key in MODELS:
            with self.subTest(model=model_key):
                cfg = _load_official(model_key)
                all_types = [
                    lp.compact_type(e) for e in lp.classify_layers(model_key, cfg)
                ]
                plan = _plan_for(model_key, cfg, 8)
                self.assertEqual(plan["type_string"].split(","), all_types[:8])

    def test_k3_shared_pool_and_glm_indexer(self):
        cfg = _load_official("kimi-k3")
        plan = _plan_for("kimi-k3", cfg, 8)
        by_name = {g["name"]: g for g in plan["kv_groups"]}
        self.assertEqual(by_name["mla"]["shared_pool"], "k3_mixed_pool")
        self.assertEqual(by_name["kda"]["shared_pool"], "k3_mixed_pool")
        self.assertEqual(by_name["kda"]["kind"], "snapshot")

        cfg = _load_official("glm-5.3")
        plan = _plan_for("glm-5.3", cfg, 8)
        by_name = {g["name"]: g for g in plan["kv_groups"]}
        self.assertEqual(by_name["indexer"]["params"]["index_topk"], 2048)
        self.assertEqual(by_name["dsa"]["kind"], "chain")

    def test_dsv4_swa_branch_and_indexer_sidecar(self):
        cfg = _load_official("deepseek-v4")
        plan = _plan_for("deepseek-v4", cfg, 8)
        by_name = {g["name"]: g for g in plan["kv_groups"]}
        swa = by_name["swa"]
        self.assertEqual(swa["params"]["sliding_window"], 128)
        self.assertTrue(swa["params"]["per_layer_branch"])
        self.assertEqual(swa["layers"], list(range(8)))
        idx = by_name["indexer"]
        self.assertEqual(idx["kind"], "sidecar")
        # 8 层时层 2..7 全是 CSA(C4/C128 交替),indexer 跟随全部 CSA 源组
        self.assertEqual(idx["layers"], [2, 3, 4, 5, 6, 7])


class TestTemplateFallback(unittest.TestCase):
    """离线无 config 时,报告口径模板必须能独立产出同样的架构形状。"""

    def test_template_patterns(self):
        for model_key in MODELS:
            with self.subTest(model=model_key):
                cfg = lp.template_config(model_key)
                full = lp.classify_layers(model_key, cfg)
                self.assertEqual(len(full), EXPECTED_TOTAL[model_key])
                type_str = ",".join(lp.compact_type(e) for e in full[:8])
                self.assertEqual(type_str, EXPECTED_FIRST8[model_key])

    def test_template_kv_shape_snapshot_nonempty(self):
        for model_key in MODELS:
            with self.subTest(model=model_key):
                cfg = lp.template_config(model_key)
                snap = lp.kv_shape_snapshot(model_key, cfg)
                self.assertTrue(snap, f"{model_key}: 模板缺少 KV 形状快照字段")


class TestReducedConfig(unittest.TestCase):
    """(a)/(c):层数、KV 形状保留。"""

    def test_layer_count(self):
        for model_key in MODELS:
            with self.subTest(model=model_key):
                cfg = _load_official(model_key)
                reduced = lp.reduce_config(model_key, cfg, 8)
                if model_key == "deepseek-v4":
                    self.assertEqual(reduced["num_hidden_layers"], 8)
                else:
                    self.assertEqual(reduced["text_config"]["num_hidden_layers"], 8)

    def test_kv_shape_fields_preserved(self):
        """(c) 缩减前后所有 KV 关键形状标量必须一字不差。"""
        for model_key in MODELS:
            with self.subTest(model=model_key):
                cfg = _load_official(model_key)
                reduced = lp.reduce_config(model_key, cfg, 8)
                snap_orig = lp.kv_shape_snapshot(model_key, cfg)
                snap_red = lp.kv_shape_snapshot(model_key, reduced)
                self.assertEqual(snap_red, snap_orig)

    def test_compress_ratios_prefix_preserved(self):
        cfg = _load_official("deepseek-v4")
        reduced = lp.reduce_config("deepseek-v4", cfg, 8)
        self.assertEqual(reduced["compress_ratios"], cfg["compress_ratios"][:8])

    def test_ffn_shrunk_cap(self):
        cfg = _load_official("deepseek-v4")
        reduced = lp.reduce_config("deepseek-v4", cfg, 8)
        self.assertLess(reduced["n_routed_experts"], cfg["n_routed_experts"])
        self.assertLessEqual(
            reduced["num_experts_per_tok"], reduced["n_routed_experts"]
        )
        kept = lp.reduce_config("deepseek-v4", cfg, 8, shrink_ffn=False)
        self.assertEqual(kept["n_routed_experts"], cfg["n_routed_experts"])


class TestLayerPlanJson(unittest.TestCase):
    """(d) layer_plan.json 结构合法性。"""

    def _valid_plan(self, model_key: str, n: int = 8) -> dict:
        cfg = _load_official(model_key)
        source = {
            "kind": "official_config",
            "repo": lp.OFFICIAL[model_key][0],
            "url": lp.OFFICIAL[model_key][1],
            "local_file": "official_configs/" + model_key + ".json",
            "sha256": lp.sha256_of_config(cfg),
        }
        return lp.build_layer_plan(model_key, cfg, n, source)

    def _assert_structure(self, plan: dict) -> None:
        self.assertEqual(plan["schema_version"], 1)
        for key in (
            "model_key",
            "official_repo",
            "source",
            "num_hidden_layers_original",
            "num_layers_requested",
            "num_layers",
            "cfg_kv_shape",
            "layer_plan",
            "kv_groups",
        ):
            self.assertIn(key, plan)
        self.assertIn(plan["source"]["kind"], ("official_config", "template"))
        if plan["source"]["kind"] == "official_config":
            self.assertTrue(plan["source"].get("url"))

        entries = plan["layer_plan"]
        self.assertEqual(len(entries), plan["num_layers"])
        self.assertEqual([e["index"] for e in entries], list(range(plan["num_layers"])))
        for e in entries:
            self.assertIn(e["type"], lp.ALL_KINDS)
            for key in ("group", "mlp", "sliding_window", "params"):
                self.assertIn(key, e)

        names = [g["name"] for g in plan["kv_groups"]]
        self.assertEqual(len(names), len(set(names)), "KV 组名必须唯一")
        for g in plan["kv_groups"]:
            self.assertIn(g["kind"], ("chain", "snapshot", "sidecar"))
            if g["kind"] in ("chain", "snapshot"):
                self.assertGreater(g["block_size"], 0)
            self.assertTrue(all(0 <= i < plan["num_layers"] for i in g["layers"]))
            self.assertTrue(g["seed"])

    def test_structure_official(self):
        for model_key in MODELS:
            with self.subTest(model=model_key):
                self._assert_structure(self._valid_plan(model_key, 8))

    def test_group_layers_cover_entries(self):
        """组可能合法重叠(SWA 每层分支 / indexer 侧车),但并集必须覆盖全部层。"""
        plan = self._valid_plan("deepseek-v4", 8)
        covered = {i for g in plan["kv_groups"] for i in g["layers"]}
        self.assertEqual(sorted(covered), list(range(8)))

    def test_structure_template(self):
        for model_key in MODELS:
            with self.subTest(model=model_key):
                cfg = lp.template_config(model_key)
                plan = _plan_for(model_key, cfg, 8)
                self._assert_structure(plan)
                self.assertEqual(plan["source"]["kind"], "template")


class TestCliRuns(unittest.TestCase):
    """CLI 研究模式、--out 模式、自检。"""

    def test_research_mode_prints_json(self):
        for model_key in MODELS:
            with self.subTest(model=model_key):
                rc, text = _run_cli(["--model", model_key, "--layers", "8"])
                self.assertEqual(rc, 0)
                plan = json.loads(text)  # 打印的就是 layer_plan
                self.assertEqual(plan["model_key"], model_key)
                self.assertEqual(plan["num_layers"], 8)

    def test_out_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model"
            rc, text = _run_cli(
                ["--model", "deepseek-v4", "--layers", "8", "--out", str(out)]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out / "layer_plan.json").exists())
            self.assertTrue((out / "config.json").exists())
            self.assertTrue((out / "official_config.json").exists())
            plan = json.loads((out / "layer_plan.json").read_text(encoding="utf-8"))
            reduced = json.loads((out / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(reduced["num_hidden_layers"], 8)
            self.assertEqual(plan["type_string"], EXPECTED_FIRST8["deepseek-v4"])

    def test_weights_flag_writes_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model"
            rc, _ = _run_cli(
                [
                    "--model",
                    "kimi-k3",
                    "--layers",
                    "4",
                    "--out",
                    str(out),
                    "--weights",
                    "--seed",
                    "7",
                    "--shrink-vocab",
                    "4096",
                ]
            )
            self.assertEqual(rc, 0)
            wdir = out / "weights"
            self.assertTrue((wdir / "model.safetensors").exists())

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            lp.resolve_model_key("qwen-not-a-real-model")


class TestFakeWeights(unittest.TestCase):
    """(g) 纯 stdlib 的 safetensors 随机权重:清单 -> 写 -> 读回。"""

    def test_manifest_and_roundtrip(self):
        cfg = _load_official("deepseek-v4")
        plan = _plan_for("deepseek-v4", cfg, 4)
        reduced = lp.reduce_config("deepseek-v4", cfg, 4, shrink_vocab=4096)
        manifest = fake_weights.build_manifest(plan, reduced)
        self.assertTrue(manifest)
        names = {t["name"] for t in manifest}
        self.assertIn("model.embed_tokens.weight", names)
        self.assertTrue(any("self_attn.q_proj.weight" in n for n in names))

        with tempfile.TemporaryDirectory() as tmp:
            written = fake_weights.write_safetensors(
                manifest, Path(tmp), seed=1234, shards=2
            )
            self.assertEqual(len(written), 3)  # 2 分片 + index.json
            header, data_len = fake_weights.read_safetensors_header(written[0])
            with open(written[0], "rb") as fh:
                header_len = struct.unpack("<Q", fh.read(8))[0]
                fh.seek(0, os.SEEK_END)
                file_size = fh.tell()
            self.assertEqual(file_size, 8 + header_len + data_len)
            for name, t in header.items():
                if name == "metadata":
                    continue
                self.assertEqual(t["dtype"], "F32")
                s, e = t["data_offsets"]
                self.assertLess(s, e)
                self.assertLessEqual(e, file_size)
                n_elem = 1
                for dim in t["shape"]:
                    n_elem *= dim
                self.assertEqual(e - s, n_elem * 4)


class TestWeightsManifestShapes(unittest.TestCase):
    def test_mla_weights_shapes(self):
        cfg = _load_official("kimi-k3")
        plan = _plan_for("kimi-k3", cfg, 4)
        reduced = lp.reduce_config("kimi-k3", cfg, 4)
        manifest = fake_weights.build_manifest(plan, reduced)
        by_name = {t["name"]: t["shape"] for t in manifest}
        hidden = 7168
        self.assertEqual(
            by_name["model.embed_tokens.weight"],
            [reduced["text_config"]["vocab_size"], hidden],
        )
        kv_a = by_name["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]
        self.assertEqual(kv_a, [512 + 64, hidden])  # kv_lora_rank + qk_rope_head_dim
        # 4 层 plan:层 0 是 MLA(dense replace 层)
        o = by_name["model.layers.0.self_attn.o_proj.weight"]
        self.assertEqual(o, [hidden, 96 * 128])  # num_heads * head_dim

    def test_csa_compressed_shapes(self):
        cfg = _load_official("deepseek-v4")
        plan = _plan_for("deepseek-v4", cfg, 8)
        reduced = lp.reduce_config("deepseek-v4", cfg, 8)
        manifest = fake_weights.build_manifest(plan, reduced)
        by_name = {t["name"]: t["shape"] for t in manifest}
        ck = by_name["model.layers.2.self_attn.compress_k_proj.weight"]  # 层 2 = C4
        self.assertEqual(ck, [64 * 512 // 4, 4096])
        idx = by_name["model.layers.2.self_attn.indexer_k_proj.weight"]
        self.assertEqual(idx, [64 * 128, 4096])  # index_n_heads * index_head_dim


if __name__ == "__main__":
    unittest.main(verbosity=2)
