#!/usr/bin/env python3
"""可选:离线随机权重 safetensors 生成器(纯 stdlib,不需要 torch/numpy)。

背景:
    vllm 的 ``--load-format dummy`` 会在启动时自行生成随机权重,本模块并不是给
    vllm 用的必要件——它是"可选路径 (b)"(见 README):在离线/想要"真文件"时,
    按 layer_plan + 缩减 config 生成一包随机张量。

重要限制(README 也有说明):
    * 张量名/形状按常见 HF 命名约定 + layer_plan 里的注意力种类"启发式"生成,
      **不保证**与 vllm 各模型实现(DeepseekV4/KimiLinear/GLM5)的显式参数名
      一一对应;不要拿它喂 ``--load-format safetensors`` 期望一定能加载。
    * 用途定位:给 UCM 自己的验证代码(规格表/双原语/形状解析)提供"像真的
      权重文件"。推理请始终走 ``--load-format dummy``。
    * 数值是 [−0.02, 0.02] 的均匀随机;有 numpy 时用 numpy 提速,没有则纯
      Python ``array`` 兜底(大张量会慢,可配合 ``--shrink-vocab`` 减量)。

用法:
    # 由 build_fake_model.py 顺带生成(--weights)
    python build_fake_model.py --model deepseek-v4 --layers 8 --out ./fake --weights

    # 或独立使用
    python fake_weights.py --plan ./fake/layer_plan.json --config ./fake/config.json \
        --out ./fake/weights --seed 1234 --shards 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_F32 = "F32"
_F32_BYTES = 4
_ALIGN = 8  # 每个张量数据 8 字节对齐(足够被 safetensors 读取器接受)

_NAME_ATTN_TENSORS = {
    # 注意力种类 -> 张量名模板(占位符 {i} = 层号)
    "full": [
        "model.layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight",
    ],
    "mla": [
        "model.layers.{i}.self_attn.q_a_proj.weight",
        "model.layers.{i}.self_attn.q_b_proj.weight",
        "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.{i}.self_attn.kv_b_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight",
    ],
    "csa": [
        "model.layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.compress_k_proj.weight",
        "model.layers.{i}.self_attn.compress_v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight",
    ],
    "kda": [
        "model.layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight",
        "model.layers.{i}.self_attn.short_conv1d.weight",
        "model.layers.{i}.self_attn.gate_proj.weight",
    ],
    "dsa": [
        "model.layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight",
    ],
    "dspark": [
        "model.layers.{i}.self_attn.q_proj.weight",
        "model.layers.{i}.self_attn.k_proj.weight",
        "model.layers.{i}.self_attn.v_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight",
        "model.layers.{i}.self_attn.hash_proj.weight",
    ],
}


def _text(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """取"文本级"配置:DSV4 是平铺,K3/GLM 在 text_config 里。"""
    return cfg.get("text_config", cfg)


def _v(cfg: Dict[str, Any], name: str, default: Any = None) -> Any:
    val = cfg.get(name, default)
    return default if val is None else val


# ---------------------------------------------------------------------------
# 清单构建
# ---------------------------------------------------------------------------
def build_manifest(
    plan: Dict[str, Any], reduced_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """按 layer_plan + 缩减 config 生成 (name, shape) 清单。

    返回 list[{"name", "shape", "dtype"}],供 write_safetensors 使用。
    """
    cfg = _text(reduced_config)
    model_key = plan["model_key"]
    hidden = int(_v(cfg, "hidden_size", 4096))
    n_heads = int(_v(cfg, "num_attention_heads", 0) or 0)
    n_kv_heads = int(_v(cfg, "num_key_value_heads", n_heads) or n_heads)
    head_dim = int(
        _v(
            cfg,
            "head_dim",
            _v(
                cfg.get("linear_attn_config", {}),
                "head_dim",
                _v(cfg, "qk_head_dim", 0) or 0,
            ),
        )
        or 0
    )
    q_lora = int(_v(cfg, "q_lora_rank", 0) or 0)
    kv_lora = int(_v(cfg, "kv_lora_rank", 0) or 0)
    rope_dim = int(_v(cfg, "qk_rope_head_dim", 0) or 0)
    idx_heads = int(_v(cfg, "index_n_heads", 0) or 0)
    idx_dim = int(_v(cfg, "index_head_dim", 0) or 0)
    inter = int(_v(cfg, "intermediate_size", 0) or 0)
    moe_inter = int(_v(cfg, "moe_intermediate_size", 0) or inter)
    n_routed = int(_v(cfg, "n_routed_experts", _v(cfg, "num_experts", 0) or 0) or 0)
    n_shared = int(
        _v(cfg, "n_shared_experts", _v(cfg, "num_shared_experts", 1) or 1) or 1
    )
    vocab = int(_v(cfg, "vocab_size", 0) or 0)
    tie = bool(_v(cfg, "tie_word_embeddings", False))

    manifest: List[Dict[str, Any]] = []
    add = lambda name, *shape: manifest.append(
        {"name": name, "shape": [int(s) for s in shape], "dtype": _F32}
    )

    if vocab:
        add("model.embed_tokens.weight", vocab, hidden)
    if not tie and vocab:
        add("lm_head.weight", vocab, hidden)

    for entry in plan["layer_plans"] if "layer_plans" in plan else plan["layer_plan"]:
        i = entry["index"]
        kind = entry["type"]
        params: Dict[str, Any] = entry.get("params") or {}
        h_dim = _pick(params, head_dim, "head_dim")
        h_heads = _pick(params, n_heads, "num_heads")

        add(f"model.layers.{i}.input_layernorm.weight", hidden)
        for tpl in _NAME_ATTN_TENSORS.get(kind, _NAME_ATTN_TENSORS["full"]):
            name = tpl.format(i=i)
            shape = _attn_shape(
                name,
                kind,
                params,
                hidden=hidden,
                n_heads=h_heads,
                n_kv_heads=n_kv_heads,
                head_dim=h_dim,
                q_lora=q_lora,
                kv_lora=kv_lora,
                rope_dim=rope_dim,
                idx_heads=idx_heads,
                idx_dim=idx_dim,
            )
            if shape:
                add(name, *shape)

        # 索引器 sidecar 探针(跟随 CSA/DSA 源组)
        if kind in ("csa", "dsa") and idx_heads and idx_dim:
            add(
                f"model.layers.{i}.self_attn.indexer_k_proj.weight",
                idx_heads * idx_dim,
                hidden,
            )

        # MLP
        if entry.get("mlp") == "moe" and n_routed > 0:
            add(f"model.layers.{i}.mlp.gate.weight", n_routed, hidden)
            add(
                f"model.layers.{i}.mlp.experts.gate_up_proj.weight",
                n_routed,
                2 * moe_inter,
                hidden,
            )
            add(
                f"model.layers.{i}.mlp.experts.down_proj.weight",
                n_routed,
                moe_inter,
                hidden,
            )
            if n_shared > 0:
                add(
                    f"model.layers.{i}.mlp.shared_experts.gate_up_proj.weight",
                    n_shared * 2 * moe_inter,
                    hidden,
                )
                add(
                    f"model.layers.{i}.mlp.shared_experts.down_proj.weight",
                    n_shared * moe_inter,
                    hidden,
                )
        else:
            if not inter:
                inter = moe_inter
            add(f"model.layers.{i}.mlp.gate_proj.weight", inter, hidden)
            add(f"model.layers.{i}.mlp.up_proj.weight", inter, hidden)
            add(f"model.layers.{i}.mlp.down_proj.weight", hidden, inter)

        add(f"model.layers.{i}.post_attention_layernorm.weight", hidden)

    add("model.norm.weight", hidden)
    return manifest


def _pick(params: Dict[str, Any], fallback: int, name: str) -> int:
    val = params.get(name)
    return int(val) if val else fallback


def _attn_shape(
    name: str,
    kind: str,
    params: Dict[str, Any],
    hidden: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    q_lora: int,
    kv_lora: int,
    rope_dim: int,
    idx_heads: int,
    idx_dim: int,
) -> Optional[Tuple[int, ...]]:
    """按张量名给出形状;拿不准的返回 None(跳过)。"""
    base = name.rsplit(".", 2)[1]
    ratio = int(params.get("compress_ratio") or 0) or None

    if base == "q_proj":
        return (n_heads * head_dim, hidden)
    if base == "k_proj":
        return (n_kv_heads * head_dim, hidden)
    if base == "v_proj":
        return (n_kv_heads * head_dim, hidden)
    if base == "o_proj":
        return (hidden, n_heads * head_dim)
    if base == "q_a_proj" and q_lora:
        return (q_lora, hidden)
    if base == "q_b_proj" and q_lora:
        return (n_heads * head_dim, q_lora)
    if base == "kv_a_proj_with_mqa" and kv_lora:
        return (kv_lora + rope_dim, hidden)
    if base == "kv_b_proj" and kv_lora:
        return (n_heads * head_dim, kv_lora + rope_dim)
    if base == "compress_k_proj" and ratio:
        return (max(1, n_heads * head_dim // ratio), hidden)
    if base == "compress_v_proj" and ratio:
        return (max(1, n_heads * head_dim // ratio), hidden)
    if base == "short_conv1d":
        ks = int(params.get("short_conv_kernel_size") or 4)
        return (n_heads * head_dim, n_heads * head_dim * ks)
    if base == "gate_proj":
        return (n_heads * head_dim, n_heads * head_dim)
    if base == "hash_proj":
        return (hidden, hidden)
    if base == "indexer_k_proj" and idx_heads and idx_dim:
        return (idx_heads * idx_dim, hidden)
    return None


# ---------------------------------------------------------------------------
# safetensors 写入(纯 stdlib)
# ---------------------------------------------------------------------------
_CHUNK = 1 << 21  # 每块 200 万元素(8 MB),避免大张量一次性占满内存


def _fill_tensor_stream(n_elem: int, seed: int, offset: int):
    """分块流式生成 F32 随机字节,返回可迭代的 bytes 块。"""
    rng_seed = seed * 1000003 + offset
    rng = random.Random(rng_seed)
    try:
        import numpy as np  # type: ignore

        rng_np = np.random.default_rng(rng_seed)
        remaining = n_elem
        while remaining > 0:
            take = min(remaining, _CHUNK)
            arr = rng_np.uniform(-0.02, 0.02, size=take)
            yield arr.astype("<f4").tobytes()
            remaining -= take
    except ImportError:
        from array import array

        remaining = n_elem
        while remaining > 0:
            take = min(remaining, _CHUNK)
            chunk = array("f", (rng.uniform(-0.02, 0.02) for _ in range(take)))
            yield chunk.tobytes()
            remaining -= take


def write_safetensors(
    manifest: Sequence[Dict[str, Any]],
    out_dir: Path,
    seed: int = 1234,
    shards: int = 1,
) -> List[Path]:
    """把清单写成一包 safetensors(little-endian F32,8 字节对齐)。

    data_offsets 是文件内绝对偏移(与 safetensors 规范一致):
    数据区起点 = 8 字节长度头 + header JSON 长度,而 header 长度又依赖
    data_offsets 的数字位数 —— 这里用定点迭代收敛(1~2 轮即稳定)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = max(1, min(int(shards), len(manifest) or 1))

    chunks: List[List[Dict[str, Any]]] = [[] for _ in range(shards)]
    for idx, t in enumerate(manifest):
        chunks[idx % shards].append(t)

    written: List[Path] = []
    weight_map: Dict[str, str] = {}
    for shard_no, tensors in enumerate(chunks):
        if not tensors:
            continue
        if shards == 1:
            fname = "model.safetensors"
        else:
            fname = f"model-{shard_no}-of-{shards}.safetensors"
        path = out_dir / fname

        # 1) 数据区布局(纯算术,不物化数据:size = 4*prod(shape))
        #    data_offsets = [数据起点, 数据终点](不含对齐填充);下一个张量从填充后开始
        records: List[Tuple[str, List[int], int]] = []  # (name, shape, size_bytes)
        rel_offsets: List[Tuple[int, int]] = []  # (rs, re) 数据区相对范围
        cursor = 0
        for t in tensors:
            size = _F32_BYTES * (math.prod(t["shape"]) if t["shape"] else 0)
            start = cursor
            cursor += size
            pad = (_ALIGN - (cursor % _ALIGN)) % _ALIGN
            cursor += pad
            records.append((t["name"], t["shape"], size))
            rel_offsets.append((start, start + size))
            weight_map[t["name"]] = fname

        # 2) header 定点迭代:data_offsets(绝对) <-> header 长度互相依赖
        base = 8  # 起始假设:header 长 0
        for _ in range(4):
            header_dict: Dict[str, Any] = {"metadata": {}}
            for (name, shape, _size), (rs, re) in zip(records, rel_offsets):
                header_dict[name] = {
                    "dtype": _F32,
                    "shape": list(shape),
                    "data_offsets": [base + rs, base + re],
                }
            header = json.dumps(header_dict).encode("utf-8")
            new_base = 8 + len(header)
            if new_base == base:
                break
            base = new_base

        # 3) 顺序写盘:[长度头][header][数据(分块生成,张量间补对齐填充)]
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(header)))
            fh.write(header)
            for (name, _shape, size), (rs, _re_rel) in zip(records, rel_offsets):
                jitter = sum(ord(ch) for ch in _stable_index((str(seed), name)))
                for chunk in _fill_tensor_stream(size // _F32_BYTES, seed, jitter):
                    fh.write(chunk)
                cursor_now = fh.tell() - (8 + len(header))
                pad = (_ALIGN - (cursor_now % _ALIGN)) % _ALIGN
                if pad:
                    fh.write(b"\x00" * pad)
        written.append(path)

    if shards > 1:
        index_path = out_dir / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
        written.append(index_path)
    return written


def _stable_index(record_id: Tuple[str, str]) -> str:
    """给 (seed, tensor名) 一个稳定的哈希序号,保证同种子输出可复现。"""
    import hashlib

    return hashlib.sha1(record_id[1].encode("utf-8")).hexdigest()


def generate_for_plan(
    plan: Dict[str, Any],
    reduced_config: Dict[str, Any],
    out_dir: Path,
    seed: int = 1234,
    shards: int = 1,
) -> List[Path]:
    manifest = build_manifest(plan, reduced_config)
    return write_safetensors(manifest, out_dir, seed=seed, shards=shards)


# ---------------------------------------------------------------------------
# 读回校验(也作为测试工具)
# ---------------------------------------------------------------------------
def read_safetensors_header(path: Path) -> Tuple[Dict[str, Any], int]:
    """读回 safetensors 头部;返回 (header, 数据区字节数)。"""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        data = fh.read()
    return header, len(data)


# ---------------------------------------------------------------------------
# 独立 CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fake_weights.py",
        description="按 layer_plan + 缩减 config 生成随机权重 safetensors(纯 stdlib)。",
    )
    parser.add_argument("--plan", type=Path, required=True, help="layer_plan.json 路径")
    parser.add_argument("--config", type=Path, required=True, help="缩减 config.json 路径")
    parser.add_argument("--out", type=Path, default=Path("weights"), help="输出目录")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shards", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_manifest(plan, cfg)
    written = write_safetensors(manifest, args.out, seed=args.seed, shards=args.shards)
    print(f"manifest tensors: {len(manifest)}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
