#!/usr/bin/env python3
"""假模型生成器:按官方 config 的层分布/注意力类型,产出可在单卡 NPU 上跑的缩减模型。

动机(详见同目录 README.md):
    目标模型(DeepSeek-V4-Flash、Kimi K3、GLM-5.3-Flash)权重数十~上千 GB,单张
    64GB NPU 装不下;UCM 只需要"真实架构的 KV 组结构"(FULL/MLA/CSA/SWA/KDA/DSA/
    Indexer)去验证规格表与双原语。做法:保留官方 config.json 的层分布与 KV 形状,
    把层数砍到 N(默认 8),配合 vllm 的 ``--load-format dummy`` 随机初始化权重,
    不下载任何真实权重文件。

用法:
    # 研究模式:只打印 layer_plan(不写文件)
    python build_fake_model.py --model deepseek-v4 --layers 8

    # 生成缩减模型目录(dummy 加载即可跑)
    python build_fake_model.py --model deepseek-v4 --layers 8 --out ./fake_dsv4
    vllm serve ./fake_dsv4/config.json --load-format dummy ...

    # 离线随机权重 safetensors(可选,见 fake_weights.py)
    python build_fake_model.py --model kimi-k3 --layers 8 --out ./fake_k3 --weights

依赖: python3 标准库;requests/torch 均为可选(缺 requests 自动退回 urllib,
缺 torch 不影响任何功能——safetensors 生成是纯 stdlib 实现)。

本脚本为自包含模块,永不 import ucm 包(见 README"无污染"说明)。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from layer_plans import (
    OFFICIAL,
    build_layer_plan,
    kv_shape_snapshot,
    reduce_config,
    resolve_model_key,
    sha256_of_config,
    template_config,
    template_source,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_OFFICIAL_DIR = _SCRIPT_DIR / "official_configs"
_UA = "UCM-fake-model-builder/1.0 (+https://github.com/ModelEngine-Group/unified-cache-management)"

# 官方 config 抓取兜底镜像(huggingface.co 连不上时自动换 hf-mirror.com)
_HF_MIRROR_PREFIX = "https://hf-mirror.com"


class FetchError(RuntimeError):
    """抓取官方 config 失败(网络 / 404 / 解析错误)。"""


def _mirror_url(url: str) -> Optional[str]:
    """把 huggingface.co URL 改写成 hf-mirror.com 镜像,非 HF 地址返回 None。"""
    if url.startswith("https://huggingface.co/"):
        return _HF_MIRROR_PREFIX + url[len("https://huggingface.co") :]
    if url.startswith("https://hf-mirror.com/"):
        return url
    return None


# ---------------------------------------------------------------------------
# 来源加载:缓存官方 config > 网络抓取(HF → hf-mirror 兜底)> 报告口径模板
# ---------------------------------------------------------------------------
def _fetch_hf_config(url: str, timeout: int = 30) -> Dict[str, Any]:
    """抓取 HF 原始 config.json。优先 requests(可选依赖),退回 urllib。"""
    try:  # requests 是可选依赖
        import requests  # type: ignore

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": _UA})
        resp.raise_for_status()
        return dict(resp.json())
    except ImportError:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                raw = fh.read().decode("utf-8")
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
        ) as exc:
            raise FetchError(f"抓取 {url} 失败: {exc}") from exc
        try:
            return dict(json.loads(raw))
        except ValueError as exc:
            raise FetchError(f"{url} 不是合法 JSON: {exc}") from exc


def fetch_with_mirror(url: str, timeout: int = 30) -> Tuple[Dict[str, Any], bool]:
    """抓取官方 config;HF 主站失败时自动换 hf-mirror.com,返回 (config, 是否走镜像)。"""
    try:
        return _fetch_hf_config(url, timeout=timeout), False
    except FetchError as primary_err:
        mirror = _mirror_url(url)
        if mirror is None:
            raise
        try:
            return _fetch_hf_config(mirror, timeout=timeout), True
        except FetchError as mirror_err:
            raise FetchError(f"主站与镜像均失败: {primary_err}; {mirror_err}") from mirror_err


def load_source_config(
    model_key: str, fetch: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """加载源 config,返回 (config, source 元信息)。

    优先级: 缓存官方 config(official_configs/) > [--fetch] 网络 > 模板。
    """
    repo, url, fname = OFFICIAL[model_key]
    local = _OFFICIAL_DIR / fname

    if local.exists() and not fetch:
        cfg = json.loads(local.read_text(encoding="utf-8"))
        source = {
            "kind": "official_config",
            "repo": repo,
            "url": url,
            "local_file": f"official_configs/{fname}",
            "sha256": sha256_of_config(cfg),
            "note": "仓库内缓存的官方 config.json(2026 官方发布版)",
        }
        return cfg, source

    # 网络抓取(显式 --fetch,或缓存缺失时自动尝试;失败静默退化)
    if fetch or not local.exists():
        try:
            cfg, via_mirror = fetch_with_mirror(url)
            if not local.exists() or fetch:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(
                    json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            source = {
                "kind": "official_config",
                "repo": repo,
                "url": url,
                "local_file": f"official_configs/{fname}",
                "sha256": sha256_of_config(cfg),
                "note": (
                    "本次从 HuggingFace 官方仓库抓取"
                    if not via_mirror
                    else "本次经 hf-mirror.com 镜像抓取(主站不可达)"
                ),
                "fetched": True,
                "fetched_via": "hf-mirror.com" if via_mirror else "huggingface.co",
            }
            return cfg, source
        except FetchError as exc:
            print(f"[warn] {exc};退回报告口径模板", file=sys.stderr)

    return template_config(model_key), template_source(model_key)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _plan_summary(model_key: str, plan: Dict[str, Any]) -> str:
    src = plan["source"]
    kind = src.get("kind")
    url = src.get("url") or "(模板)"
    groups = ", ".join(g["name"] for g in plan["kv_groups"])
    return (
        f"model             : {model_key} (官方仓库 {plan['official_repo']})\n"
        f"source            : {kind} {url}\n"
        f"layers            : original {plan['num_hidden_layers_original']} -> "
        f"{plan['num_layers']} (requested {plan['num_layers_requested']})\n"
        f"layer types       : {plan['type_string']}\n"
        f"kv groups         : {groups}\n"
    )


def write_outputs(
    model_key: str,
    plan: Dict[str, Any],
    reduced: Dict[str, Any],
    original: Dict[str, Any],
    out_dir: Path,
    weights: bool = False,
    seed: int = 1234,
    shards: int = 1,
) -> List[Path]:
    """把 layer_plan.json / config.json / official_config.json 写入 out_dir。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    plan_path = out_dir / "layer_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written.append(plan_path)

    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(reduced, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written.append(config_path)

    orig_path = out_dir / "official_config.json"
    orig_path.write_text(
        json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written.append(orig_path)

    if weights:
        try:
            import fake_weights  # 本地模块,纯 stdlib
        except ImportError as exc:  # pragma: no cover - 防御
            print(
                f"[warn] 无法 import fake_weights({exc}),跳过权重生成",
                file=sys.stderr,
            )
        else:
            weights_dir = out_dir / "weights"
            written.extend(
                fake_weights.generate_for_plan(plan, reduced, weights_dir, seed, shards)
            )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_fake_model.py",
        description="按官方 config 生成缩减层数的假模型(config + layer_plan)。",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="目标模型: deepseek-v4 / kimi-k3 / glm-5.3(支持常见别名)。",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=8,
        help="保留的层数(默认 8;从第 0 层起截断,保留原始层类型模式)。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录(写 layer_plan.json/config.json/official_config.json)。"
        "缺省 = 研究模式,只打印 layer_plan。",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="强制从 HuggingFace 官方仓库抓取 config(失败自动退化)。",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="强制使用报告 6.1 口径内置模板(跳过缓存与网络)。",
    )
    parser.add_argument(
        "--keep-ffn",
        action="store_true",
        help="不缩小 MoE/FFN 字段(默认缩小以降低 dummy 权重显存,不影响 KV 形状)。",
    )
    parser.add_argument(
        "--shrink-vocab",
        type=int,
        default=0,
        metavar="N",
        help="把 vocab_size 缩到 N(0=保持官方值)。非 KV 字段,可选省显存。",
    )
    parser.add_argument(
        "--drop-vision",
        action="store_true",
        help="去掉 vision_config(多模态模型如 K3/GLM 可选,省 vision tower 显存)。",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="同时生成随机权重的 safetensors(需 --out;启发式清单,见 fake_weights.py)。",
    )
    parser.add_argument("--seed", type=int, default=1234, help="随机权重种子(默认 1234)。")
    parser.add_argument("--shards", type=int, default=1, help="safetensors 分片数(默认 1)。")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    model_key = resolve_model_key(args.model)

    if args.template:
        config, source = template_config(model_key), template_source(model_key)
    else:
        config, source = load_source_config(model_key, fetch=args.fetch)

    original = json.loads(json.dumps(config))  # 深拷贝,作为"原样保留"比对基准
    reduced = reduce_config(
        model_key,
        config,
        args.layers,
        shrink_ffn=not args.keep_ffn,
        shrink_vocab=args.shrink_vocab,
        drop_vision=args.drop_vision,
    )
    plan = build_layer_plan(model_key, config, args.layers, source)
    n = plan["num_layers"]
    if n < args.layers:
        print(
            f"[warn] 模型 {model_key} 原始只有 {plan['num_hidden_layers_original']} 层,"
            f"截断到 {n} 层",
            file=sys.stderr,
        )

    # 自检:缩减前后 KV 形状必须一致;逐层类型模式必须一致(前 N 层)
    _verify_reduced(model_key, original, reduced, plan)

    if args.out is None:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    written = write_outputs(
        model_key,
        plan,
        reduced,
        original,
        args.out,
        weights=args.weights,
        seed=args.seed,
        shards=args.shards,
    )
    print(_plan_summary(model_key, plan), end="")
    print(f"written ({len(written)} files):")
    for p in written:
        print(f"  {p}")
    if args.weights:
        print(
            "[info] 权重为启发式清单生成的随机张量;vllm 推理请用 --load-format dummy"
            "(本生成器不保证与 vllm 参数名一一对应)。",
            file=sys.stderr,
        )
    return 0


def _verify_reduced(
    model_key: str,
    original: Dict[str, Any],
    reduced: Dict[str, Any],
    plan: Dict[str, Any],
) -> None:
    """退出前的自检:KV 形状保留 + 前 N 层类型模式一致。"""
    snap_orig = kv_shape_snapshot(model_key, original)
    snap_red = kv_shape_snapshot(model_key, reduced)
    diff = {
        k: (snap_orig[k], snap_red.get(k))
        for k in snap_orig
        if snap_red.get(k) != snap_orig[k]
    }
    missing = [k for k in snap_orig if k not in snap_red]
    if diff or missing:
        raise RuntimeError(f"缩减 config 丢失/改动了 KV 形状字段: diff={diff} missing={missing}")

    n_red = reduced.get("num_hidden_layers")
    if model_key != "deepseek-v4":
        n_red = (reduced.get("text_config") or {}).get("num_hidden_layers")
    if int(n_red or 0) != plan["num_layers"]:
        raise RuntimeError("缩减 config 的 num_hidden_layers 与 layer_plan 层数不一致")


if __name__ == "__main__":
    sys.exit(main())
