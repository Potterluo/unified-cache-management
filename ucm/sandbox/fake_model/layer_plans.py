"""假模型"层分布 → KV 组种类"的解析与推导(纯 stdlib,无任何第三方依赖)。

本模块是 ``build_fake_model.py`` 的纯逻辑核心,不和 ucm 包混在一起
(``ucm/__init__.py`` 会拖入 vllm patch,见 README 的"无污染"说明):

* ``classify_layers``     —— 把官方 config 逐层翻译成注意力种类(full/mla/csa/kda/dsa/dspark)
* ``derive_kv_groups``    —— 按 UCM 报告 4.1/6.1/8.1 的口径,把"层"归并成 KV 组
                             (chain 链式 / snapshot 快照 / sidecar 侧车)
* ``reduce_config``       —— 砍层数到 N、缩小 MoE/FFN,但 KV 相关形状字段原样保留
* ``template_config``     —— 报告 6.1 口径的内置模板(抓不到真 config 时的退化路径)

三个目标模型的官方口径来源(2026-08 报告 6.1 + 官方 config.json):
* DeepSeek-V4-Flash-0731: FULL 2 层 + CSA C4/C128 交替 + SWA 每层分支 + Indexer,
  末尾 40/41/42 层为 DeepSeek Spark(hash)稀疏层;46 项 compress_ratios。
* Kimi K3:MLA 全注意力每 4 层 1 个(另首层 dense)+ KDA 线性注意力同池混合。
* GLM-5.3-Flash:34 层 KDA 线性注意力 + 11 层 DeepSeek 稀疏注意力 + MoE,
  Indexer top-k 2048。

注意:2026 年这批"报告口径"模型属于官方发布的真实 config(已缓存到
official_configs/);一切以抓到的真 config 为准,模板只是离线/断网时的兜底。
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 注意力种类词表(可扩展;未来可加 mamba/gdn/hcsa 等)
# ---------------------------------------------------------------------------
KIND_FULL = "full"  # 全注意力(未知内部形态时按 MHA 形状估算)
KIND_MLA = "mla"  # MLA 压缩 KV 全注意力
KIND_CSA = "csa"  # 压缩稀疏注意力(CSA/HCA),带 compress_ratio
KIND_KDA = "kda"  # KDA 线性注意力(逐 token 状态,KV 组为快照)
KIND_DSA = "dsa"  # DeepSeek 稀疏注意力(带索引器 sidecar)
KIND_DSPARK = "dspark"  # DeepSeek Spark 哈希稀疏层(逐 token 状态)
KIND_SWA = "swa"  # 滑动窗口注意力(独立层;DSV4 里是"每层分支"的组)
KIND_MAMBA = "mamba"  # Mamba/SSM 状态层(本批模型未用,预留)
KIND_GDN = "gdn"  # Gated DeltaNet(预留)
KIND_NONE = "none"

ALL_KINDS = frozenset(
    {
        KIND_FULL,
        KIND_MLA,
        KIND_CSA,
        KIND_KDA,
        KIND_DSA,
        KIND_DSPARK,
        KIND_SWA,
        KIND_MAMBA,
        KIND_GDN,
        KIND_NONE,
    }
)

GROUP_KIND_CHAIN = "chain"  # 积木:可拼接的块,按内容哈希寻址(BlockStore)
GROUP_KIND_SNAPSHOT = "snapshot"  # 罐头:位置键快照,块对齐 + CoW(位置键原语)
GROUP_KIND_SIDECAR = "sidecar"  # 侧车:索引器元数据,跟随源组不投票(铁律 5)
GROUP_KIND_NONE = "none"

# ---------------------------------------------------------------------------
# 模型注册表:model_key -> (官方仓库, 官方 config URL, 本地缓存文件名)
# ---------------------------------------------------------------------------
OFFICIAL: Dict[str, Tuple[str, str, str]] = {
    "deepseek-v4": (
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/main/config.json",
        "deepseek-v4.json",
    ),
    "kimi-k3": (
        "moonshotai/Kimi-K3",
        "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json",
        "kimi-k3.json",
    ),
    "glm-5.3": (
        "zai-org/GLM-5.3-Flash",
        "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json",
        "glm-5.3.json",
    ),
}

ALIASES: Dict[str, str] = {
    "deepseek-v4": "deepseek-v4",
    "deepseek_v4": "deepseek-v4",
    "deepseekv4": "deepseek-v4",
    "ds-v4": "deepseek-v4",
    "dsv4": "deepseek-v4",
    "deepseek": "deepseek-v4",
    "kimi-k3": "kimi-k3",
    "kimi_k3": "kimi-k3",
    "kimik3": "kimi-k3",
    "kimi": "kimi-k3",
    "k3": "kimi-k3",
    "glm-5.3": "glm-5.3",
    "glm_5.3": "glm-5.3",
    "glm5.3": "glm-5.3",
    "glm-5-3": "glm-5.3",
    "glm5": "glm-5.3",
    "glm": "glm-5.3",
}

# 每模型的"KV 关键形状"标量字段(缩减 config 必须原样保留;逐层分布类字段除外)。
# 存在 text_config 里的用 dotted 路径标记。
KV_SHAPE_FIELDS: Dict[str, List[str]] = {
    "deepseek-v4": [
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "qk_rope_head_dim",
        "sliding_window",
        "compress_rope_theta",
        "rope_theta",
        "rope_scaling",
        "max_position_embeddings",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "index_share_for_mtp_iteration",
        "dspark_block_size",
        "dspark_markov_rank",
        "torch_dtype",
    ],
    "kimi-k3": [
        "text_config.hidden_size",
        "text_config.num_attention_heads",
        "text_config.num_key_value_heads",
        "text_config.kv_lora_rank",
        "text_config.q_lora_rank",
        "text_config.qk_nope_head_dim",
        "text_config.qk_rope_head_dim",
        "text_config.v_head_dim",
        "text_config.head_dim",
        "text_config.linear_attn_config.head_dim",
        "text_config.linear_attn_config.num_heads",
        "text_config.linear_attn_config.short_conv_kernel_size",
        "text_config.attn_res_block_size",
        "text_config.max_position_embeddings",
        "text_config.first_k_dense_replace",
        "text_config.mla_use_nope",
        "text_config.mla_use_output_gate",
        "text_config.use_full_rank_gate",
    ],
    "glm-5.3": [
        "text_config.hidden_size",
        "text_config.num_attention_heads",
        "text_config.num_key_value_heads",
        "text_config.head_dim",
        "text_config.qk_head_dim",
        "text_config.qk_nope_head_dim",
        "text_config.qk_rope_head_dim",
        "text_config.v_head_dim",
        "text_config.q_lora_rank",
        "text_config.kv_lora_rank",
        "text_config.linear_attn_config.head_dim",
        "text_config.linear_attn_config.num_heads",
        "text_config.linear_attn_config.short_conv_kernel_size",
        "text_config.index_topk",
        "text_config.index_n_heads",
        "text_config.index_head_dim",
        "text_config.index_kpool",
        "text_config.index_kpool_compress",
        "text_config.index_kpool_always_select_tail",
        "text_config.max_position_embeddings",
        "text_config.first_k_dense_replace",
        "text_config.mla_use_nope",
    ],
}

# MoE/FFN 缩小的目标口径(仅影响 dummy 权重显存,与 KV 形状无关)。
# 每个字段: (目标值, 含义)。num_experts_per_tok 等会再被钳制到专家数以内。
FFN_SHRINK: Dict[str, Dict[str, Tuple[int, str]]] = {
    "deepseek-v4": {
        "n_routed_experts": (16, "路由专家数(原 256)"),
        "num_experts_per_tok": (4, "激活专家数/层(原 6)"),
        "n_shared_experts": (1, "共享专家数"),
        "moe_intermediate_size": (512, "MoE 中间维(原 2048)"),
    },
    "kimi-k3": {
        "num_experts": (24, "路由专家数(原 896)"),
        "num_experts_per_token": (8, "激活专家数/层(原 16)"),
        "num_shared_experts": (2, "共享专家数"),
        "moe_intermediate_size": (512, "MoE 中间维(原 3072)"),
        "intermediate_size": (4096, "dense FFN 中间维(原 33792)"),
        "routed_expert_hidden_size": (1024, "专家 FFN 隐藏维(原 3584)"),
    },
    "glm-5.3": {
        "n_routed_experts": (16, "路由专家数(原 288)"),
        "num_experts_per_tok": (4, "激活专家数/层(原 8)"),
        "n_shared_experts": (1, "共享专家数"),
        "moe_intermediate_size": (512, "MoE 中间维(原 2048)"),
        "intermediate_size": (2048, "dense FFN 中间维(原 12288)"),
    },
}


def resolve_model_key(name: str) -> str:
    """把用户输入(含别名)规整成模型 key,未知名字直接报错。"""
    key = ALIASES.get(str(name).strip().lower())
    if key is None:
        known = ", ".join(sorted(OFFICIAL))
        raise ValueError(f"未知模型 {name!r},支持: {known} (及常见别名)")
    return key


def _get_path(cfg: Dict[str, Any], dotted: str) -> Any:
    """按 dotted 路径取值('text_config.linear_attn_config.head_dim')。"""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur: Any = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def kv_shape_snapshot(model_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """取该模型全部"KV 关键形状"标量的快照(用于断言缩减前后一致)。"""
    out: Dict[str, Any] = {}
    for dotted in KV_SHAPE_FIELDS.get(model_key, []):
        value = _get_path(cfg, dotted)
        if value is not None:
            out[dotted] = value
    return out


# ---------------------------------------------------------------------------
# 逐层分类
# ---------------------------------------------------------------------------
def classify_layers(model_key: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把整个 config 翻译成逐层清单(含类型/组/mlp/滑动窗口/参数)。

    返回的每一层:
      index(int)、type(主注意力种类)、group(KV 组名)、
      mlp("moe"|"dense")、sliding_window(int|None)、params(dict)
    """
    if model_key == "deepseek-v4":
        return _classify_deepseek_v4(cfg)
    if model_key == "kimi-k3":
        return _classify_kimi_k3(cfg)
    if model_key == "glm-5.3":
        return _classify_glm5(cfg)
    raise ValueError(f"未实现的分类器: {model_key}")


def _classify_deepseek_v4(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = int(cfg["num_hidden_layers"])
    ratios = list(cfg.get("compress_ratios") or [])
    targets = {int(x) for x in (cfg.get("dspark_target_layer_ids") or [])}
    sw = cfg.get("sliding_window")
    has_moe = bool(cfg.get("n_routed_experts", 0))
    entries: List[Dict[str, Any]] = []
    for i in range(n):
        if i in targets:
            etype, group = KIND_DSPARK, "dspark"
            params = {
                "block_size": cfg.get("dspark_block_size"),
                "markov_rank": cfg.get("dspark_markov_rank"),
                "noise_token_id": cfg.get("dspark_noise_token_id"),
            }
        else:
            ratio = ratios[i] if i < len(ratios) else 0
            if not ratio:
                etype, group = KIND_FULL, "full"
                params = {
                    "head_dim": cfg.get("head_dim"),
                    "q_lora_rank": cfg.get("q_lora_rank"),
                    "o_lora_rank": cfg.get("o_lora_rank"),
                    "qk_rope_head_dim": cfg.get("qk_rope_head_dim"),
                }
            else:
                etype, group = KIND_CSA, f"csa_c{ratio}"
                params = {
                    "compress_ratio": ratio,
                    "head_dim": cfg.get("head_dim"),
                    "storage_block_size": _storage_block(cfg, ratio),
                }
        entries.append(
            {
                "index": i,
                "type": etype,
                "group": group,
                "mlp": "moe" if has_moe else "dense",
                "sliding_window": sw,
                "params": params,
            }
        )
    return entries


def _classify_kimi_k3(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = cfg["text_config"]
    n = int(text["num_hidden_layers"])
    lac = text["linear_attn_config"]
    full_ids = {int(x) for x in (lac.get("full_attn_layers") or [])}
    kda_ids = {int(x) for x in (lac.get("kda_layers") or [])}
    dense_first = int(text.get("first_k_dense_replace") or 0)
    lac_head_dim = lac.get("head_dim")
    lac_num_heads = lac.get("num_heads")
    has_moe = bool(text.get("num_experts", 0))
    entries: List[Dict[str, Any]] = []
    for i in range(n):
        if i < dense_first or i in full_ids:
            etype, group = KIND_MLA, "mla"
            params = {
                "head_dim": lac_head_dim,
                "num_heads": lac_num_heads,
                "kv_lora_rank": text.get("kv_lora_rank"),
                "q_lora_rank": text.get("q_lora_rank"),
                "qk_rope_head_dim": text.get("qk_rope_head_dim"),
            }
        else:
            etype, group = KIND_KDA, "kda"
            params = {
                "head_dim": lac_head_dim,
                "num_heads": lac_num_heads,
                "short_conv_kernel_size": lac.get("short_conv_kernel_size"),
            }
        entries.append(
            {
                "index": i,
                "type": etype,
                "group": group,
                "mlp": "moe" if has_moe else "dense",
                "sliding_window": None,
                "params": params,
            }
        )
    return entries


def _classify_glm5(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = cfg["text_config"]
    n = int(text["num_hidden_layers"])
    lt = list(text.get("layer_types") or [])
    mlp_types = list(text.get("mlp_layer_types") or [])
    idx_types = list(text.get("indexer_types") or [])
    lac = text["linear_attn_config"]
    lac_head_dim = lac.get("head_dim")
    lac_num_heads = lac.get("num_heads")
    entries: List[Dict[str, Any]] = []
    for i in range(n):
        ty = lt[i] if i < len(lt) else "linear_attention"
        if ty == "deepseek_sparse_attention":
            etype, group = KIND_DSA, "dsa"
            params = {
                "head_dim": lac_head_dim,
                "num_heads": lac_num_heads,
                "index_topk": text.get("index_topk"),
                "index_type": idx_types[i] if i < len(idx_types) else None,
            }
        else:
            etype, group = KIND_KDA, "kda"
            params = {
                "head_dim": lac_head_dim,
                "num_heads": lac_num_heads,
                "short_conv_kernel_size": lac.get("short_conv_kernel_size"),
            }
        mlp = "moe" if (i < len(mlp_types) and mlp_types[i] == "sparse") else "dense"
        entries.append(
            {
                "index": i,
                "type": etype,
                "group": group,
                "mlp": mlp,
                "sliding_window": None,
                "params": params,
            }
        )
    return entries


def _storage_block(cfg: Dict[str, Any], ratio: int) -> Optional[int]:
    """vllm-ascend: storage_block_size = block_size // compress_ratio。"""
    block = cfg.get("kernel_block_size") or cfg.get("block_size") or 128
    if not ratio:
        return None
    return max(1, int(block) // int(ratio))


# ---------------------------------------------------------------------------
# KV 组推导(UCM 4.1/8.1 口径)
# ---------------------------------------------------------------------------
def derive_kv_groups(
    model_key: str, cfg: Dict[str, Any], entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """把(可能已截断的)逐层清单归并成 KV 组。

    链式组 block_size 取 128(vllm-ascend kernel_block_size 也是 128);
    快照组 kda 逐 token(block 1),dspark 用 config 的 dspark_block_size。
    种子命名沿用报告 4.1 的 S_xxx 风格(哈希隔离)。
    """
    if model_key == "deepseek-v4":
        return _groups_deepseek_v4(cfg, entries)
    if model_key == "kimi-k3":
        return _groups_kimi_k3(cfg, entries)
    if model_key == "glm-5.3":
        return _groups_glm5(cfg, entries)
    raise ValueError(f"未实现的组推导: {model_key}")


def _idx(entries: List[Dict[str, Any]]) -> List[int]:
    return [e["index"] for e in entries]


def _by_group(entries: List[Dict[str, Any]], group: str) -> List[int]:
    return [e["index"] for e in entries if e["group"] == group]


def _g(
    name: str,
    kind: str,
    block_size: int,
    layers: List[int],
    seed: str,
    **extra: Any,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": name,
        "kind": kind,
        "block_size": block_size,
        "layers": sorted(layers),
        "seed": seed,
    }
    out.update(extra)
    return out


def _groups_deepseek_v4(
    cfg: Dict[str, Any], entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    sw = cfg.get("sliding_window")
    head_dim = cfg.get("head_dim")
    # 1) 全注意力组
    full_layers = _by_group(entries, "full")
    if full_layers:
        groups.append(
            _g(
                "full",
                GROUP_KIND_CHAIN,
                128,
                full_layers,
                "S_full",
                params={
                    "num_attention_heads": cfg.get("num_attention_heads"),
                    "head_dim": head_dim,
                    "q_lora_rank": cfg.get("q_lora_rank"),
                    "o_lora_rank": cfg.get("o_lora_rank"),
                },
                per_token_bytes=_estimate_full_bytes(cfg),
                estimate=True,
            )
        )
    # 2) CSA 压缩组(按 compress_ratio 分组:C4 / C128)
    csa_by_ratio: Dict[int, List[int]] = {}
    for e in entries:
        if e["type"] == KIND_CSA:
            ratio = int(e["params"].get("compress_ratio") or 0)
            csa_by_ratio.setdefault(ratio, []).append(e["index"])
    for ratio in sorted(csa_by_ratio):
        layers = csa_by_ratio[ratio]
        groups.append(
            _g(
                f"csa_c{ratio}",
                GROUP_KIND_CHAIN,
                128,
                layers,
                f"S_csa_c{ratio}",
                params={
                    "compress_ratio": ratio,
                    "storage_block_size": _storage_block(cfg, ratio),
                    "num_attention_heads": cfg.get("num_attention_heads"),
                    "head_dim": head_dim,
                },
                per_token_bytes=_estimate_csa_bytes(cfg, ratio),
                estimate=True,
            )
        )
    # 3) SWA:每层分支,同 spec 层共享窗口槽位(6.1"SWA 每层分支";7.3 澄清 2)
    if sw:
        groups.append(
            _g(
                "swa",
                GROUP_KIND_CHAIN,
                128,
                _idx(entries),
                "S_swa",
                params={"sliding_window": sw, "per_layer_branch": True},
            )
        )
    # 4) 索引器 sidecar:跟随 csa/dsa 源组,不参与命中投票(铁律 5)
    indexer_layers = sorted(
        e["index"]
        for e in entries
        if e["type"] in (KIND_CSA, KIND_DSA) and e["group"] != "full"
    )
    if indexer_layers:
        groups.append(
            _g(
                "indexer",
                GROUP_KIND_SIDECAR,
                0,
                indexer_layers,
                "S_indexer",
                params={
                    "index_n_heads": cfg.get("index_n_heads"),
                    "index_head_dim": cfg.get("index_head_dim"),
                    "index_topk": cfg.get("index_topk"),
                    "tokens_per_state": "==compress_ratio(CSA 源组)",
                },
            )
        )
    # 5) dspark 快照组(逐 token 状态 / 块网格)
    dsp_layers = _by_group(entries, "dspark")
    if dsp_layers:
        dsp_block = int(cfg.get("dspark_block_size") or 1)
        groups.append(
            _g(
                "dspark",
                GROUP_KIND_SNAPSHOT,
                dsp_block,
                dsp_layers,
                "S_dspark",
                params={
                    "block_size": cfg.get("dspark_block_size"),
                    "markov_rank": cfg.get("dspark_markov_rank"),
                    "noise_token_id": cfg.get("dspark_noise_token_id"),
                },
            )
        )
    return groups


def _groups_kimi_k3(
    cfg: Dict[str, Any], entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    text = cfg["text_config"]
    lac = text["linear_attn_config"]
    mla_layers = _by_group(entries, "mla")
    kda_layers = _by_group(entries, "kda")
    groups: List[Dict[str, Any]] = []
    if mla_layers:
        groups.append(
            _g(
                "mla",
                GROUP_KIND_CHAIN,
                128,
                mla_layers,
                "S_mla",
                shared_pool="k3_mixed_pool",
                params={
                    "head_dim": lac.get("head_dim"),
                    "num_heads": lac.get("num_heads"),
                    "kv_lora_rank": text.get("kv_lora_rank"),
                    "q_lora_rank": text.get("q_lora_rank"),
                    "attn_res_block_size": text.get("attn_res_block_size"),
                },
                per_token_bytes=_estimate_mla_bytes(text),
                estimate=True,
            )
        )
    if kda_layers:
        groups.append(
            _g(
                "kda",
                GROUP_KIND_SNAPSHOT,
                1,
                kda_layers,
                "S_kda",
                shared_pool="k3_mixed_pool",
                params={
                    "head_dim": lac.get("head_dim"),
                    "num_heads": lac.get("num_heads"),
                    "short_conv_kernel_size": lac.get("short_conv_kernel_size"),
                },
                per_token_bytes=_estimate_kda_bytes(text, lac),
                estimate=True,
            )
        )
    return groups


def _groups_glm5(
    cfg: Dict[str, Any], entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    text = cfg["text_config"]
    lac = text["linear_attn_config"]
    kda_layers = _by_group(entries, "kda")
    dsa_layers = _by_group(entries, "dsa")
    groups: List[Dict[str, Any]] = []
    if kda_layers:
        groups.append(
            _g(
                "kda",
                GROUP_KIND_SNAPSHOT,
                1,
                kda_layers,
                "S_kda",
                params={
                    "head_dim": lac.get("head_dim"),
                    "num_heads": lac.get("num_heads"),
                    "short_conv_kernel_size": lac.get("short_conv_kernel_size"),
                },
                per_token_bytes=_estimate_kda_bytes(text, lac),
                estimate=True,
            )
        )
    if dsa_layers:
        groups.append(
            _g(
                "dsa",
                GROUP_KIND_CHAIN,
                128,
                dsa_layers,
                "S_dsa",
                params={
                    "head_dim": text.get("qk_head_dim"),
                    "num_heads": text.get("num_attention_heads"),
                    "index_topk": text.get("index_topk"),
                },
                per_token_bytes=_estimate_dsa_bytes(text),
                estimate=True,
            )
        )
        groups.append(
            _g(
                "indexer",
                GROUP_KIND_SIDECAR,
                0,
                dsa_layers,
                "S_indexer",
                params={
                    "index_n_heads": text.get("index_n_heads"),
                    "index_head_dim": text.get("index_head_dim"),
                    "index_topk": text.get("index_topk"),
                    "index_kpool": text.get("index_kpool"),
                    "index_kpool_compress": text.get("index_kpool_compress"),
                },
            )
        )
    return groups


# ---------------------------------------------------------------------------
# 每 token 字节估算(仅做量级参考;报告 4.1 的"每层字节"也是示例值)
# ---------------------------------------------------------------------------
def _estimate_full_bytes(cfg: Dict[str, Any]) -> Optional[int]:
    heads = cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim")
    rope_dim = cfg.get("qk_rope_head_dim")
    if not heads or not head_dim:
        return None
    # MLA 形态:显存 KV 一般是"压缩潜向量 + rope 部分",这里给简化的量级上界
    return 2 * int(heads) * int(rope_dim or head_dim)


def _estimate_csa_bytes(cfg: Dict[str, Any], ratio: int) -> Optional[int]:
    heads = cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim")
    if not heads or not head_dim or not ratio:
        return None
    return 2 * int(heads) * int(head_dim) // int(ratio)


def _estimate_mla_bytes(text: Dict[str, Any]) -> Optional[int]:
    rank = text.get("kv_lora_rank")
    heads = text.get("num_attention_heads")
    rope = text.get("qk_rope_head_dim")
    if not rank:
        return None
    rope_part = int(heads or 0) * int(rope or 0) if heads else int(rope or 0)
    return 2 * (int(rank) + rope_part)


def _estimate_kda_bytes(text: Dict[str, Any], lac: Dict[str, Any]) -> Optional[int]:
    heads = lac.get("num_heads") or text.get("num_attention_heads")
    head_dim = lac.get("head_dim") or text.get("qk_head_dim")
    if not heads or not head_dim:
        return None
    return 2 * int(heads) * int(head_dim)


def _estimate_dsa_bytes(text: Dict[str, Any]) -> Optional[int]:
    heads = text.get("num_attention_heads")
    vd = text.get("v_head_dim")
    if not heads or not vd:
        return None
    return 2 * int(heads) * int(vd)


# ---------------------------------------------------------------------------
# 缩减 config
# ---------------------------------------------------------------------------
def reduce_config(
    model_key: str,
    cfg: Dict[str, Any],
    n_layers: int,
    shrink_ffn: bool = True,
    shrink_vocab: int = 0,
    drop_vision: bool = False,
) -> Dict[str, Any]:
    """把官方 config 砍到前 n_layers 层。

    * 逐层分布字段(compress_ratios / layer_types / full_attn_layers / kda_layers …)
      只保留 <n_layers 的部分,保持"前 N 层的类型模式"与原始一致;
    * KV 相关形状字段一律不改(也就是 kv_shape_snapshot 保持不变);
    * MoE/FFN 字段默认缩小(不改变 KV 分组结构,只降 dummy 权重显存);
    * ``shrink_vocab`` >0 时把词表缩到该值(非 KV 字段,可选省显存);
    * ``drop_vision`` 去掉 vision_config(多模态模型可选)。
    """
    if n_layers < 1:
        raise ValueError(f"--layers 必须 >=1,实际 {n_layers}")
    c = copy.deepcopy(cfg)
    n_orig = _original_layer_count(model_key, c)

    if model_key == "deepseek-v4":
        if n_layers < n_orig:
            c["num_hidden_layers"] = n_layers
            ratios = list(c.get("compress_ratios") or [])
            c["compress_ratios"] = ratios[:n_layers]
            c.pop("dspark_target_layer_ids", None)
            c["num_hash_layers"] = 0
        if shrink_ffn:
            _apply_ffn_shrink(c, FFN_SHRINK["deepseek-v4"])
        if shrink_vocab:
            c["vocab_size"] = shrink_vocab
    elif model_key in ("kimi-k3", "glm-5.3"):
        text = c["text_config"]
        if n_layers < n_orig:
            text["num_hidden_layers"] = n_layers
        lac = text["linear_attn_config"]
        if lac.get("full_attn_layers") is not None:
            lac["full_attn_layers"] = [
                int(x) for x in lac["full_attn_layers"] if int(x) < n_layers
            ]
        if lac.get("kda_layers") is not None:
            lac["kda_layers"] = [int(x) for x in lac["kda_layers"] if int(x) < n_layers]
        for key in ("layer_types", "mlp_layer_types", "indexer_types"):
            if text.get(key) is not None:
                text[key] = list(text[key])[:n_layers]
        if n_layers < int(text.get("first_k_dense_replace") or 0):
            text["first_k_dense_replace"] = n_layers
        if shrink_ffn:
            _apply_ffn_shrink(text, FFN_SHRINK[model_key])
        if shrink_vocab:
            text["vocab_size"] = shrink_vocab
        if drop_vision:
            c.pop("vision_config", None)
    else:
        raise ValueError(f"未实现的缩减: {model_key}")

    _clamp_ffn_consistency(c, model_key)
    return c


def _original_layer_count(model_key: str, cfg: Dict[str, Any]) -> int:
    if model_key == "deepseek-v4":
        return int(cfg.get("num_hidden_layers") or 0)
    return int(cfg.get("text_config", {}).get("num_hidden_layers") or 0)


def _apply_ffn_shrink(cfg: Dict[str, Any], targets: Dict[str, Tuple[int, str]]) -> None:
    for field, (target, _meaning) in targets.items():
        if field in cfg and isinstance(cfg[field], int):
            cfg[field] = target


def _clamp_ffn_consistency(c: Dict[str, Any], model_key: str) -> None:
    """把 per-token 专家数钳制到<=路由专家数,避免自相矛盾。"""
    pairs = {
        "deepseek-v4": ("num_experts_per_tok", "n_routed_experts"),
        "kimi-k3": ("num_experts_per_token", "num_experts"),
        "glm-5.3": ("num_experts_per_tok", "n_routed_experts"),
    }
    per_tok, routed = pairs[model_key]
    cfg = c if model_key == "deepseek-v4" else c["text_config"]
    if per_tok in cfg and routed in cfg:
        cfg[per_tok] = min(int(cfg[per_tok]), max(1, int(cfg[routed])))


# ---------------------------------------------------------------------------
# layer_plan 组装
# ---------------------------------------------------------------------------
def compact_type(entry: Dict[str, Any]) -> str:
    """简短类型串:csa 带压缩比,其余按种类名。用于 README/CLI 的"前 N 层类型串"。"""
    t = entry["type"]
    if t == KIND_CSA:
        ratio = entry.get("params", {}).get("compress_ratio")
        return f"c{ratio}" if ratio else t
    return t


def build_layer_plan(
    model_key: str,
    cfg: Dict[str, Any],
    num_layers: int,
    source: Dict[str, Any],
) -> Dict[str, Any]:
    """组装最终的 layer_plan(结构见 README;layer_plan.json 即此结构)。"""
    entries_all = classify_layers(model_key, cfg)
    entries = entries_all[:num_layers]
    groups = derive_kv_groups(model_key, cfg, entries)
    return {
        "schema_version": 1,
        "generator": "ucm.sandbox.fake_model.build_fake_model",
        "model_key": model_key,
        "official_repo": OFFICIAL[model_key][0],
        "source": source,
        "num_hidden_layers_original": len(entries_all),
        "num_layers_requested": num_layers,
        "num_layers": len(entries),
        "cfg_kv_shape": kv_shape_snapshot(model_key, cfg),
        "layer_plan": entries,
        "kv_groups": groups,
        "type_string": ",".join(compact_type(e) for e in entries),
    }


# ---------------------------------------------------------------------------
# 报告口径模板(抓不到真 config 时的兜底;数字来自报告 6.1/4.1)
# ---------------------------------------------------------------------------
def template_config(model_key: str) -> Dict[str, Any]:
    """返回"报告口径"的模板 config(结构与官方一致,可在离线时独立工作)。"""
    builders = {
        "deepseek-v4": _template_deepseek_v4,
        "kimi-k3": _template_kimi_k3,
        "glm-5.3": _template_glm5,
    }
    builder = builders.get(model_key)
    if builder is None:
        raise ValueError(f"无模板: {model_key}")
    return builder()


def _template_deepseek_v4() -> Dict[str, Any]:
    n = 43
    # 前 2 层 FULL,2..42 交替 C4/C128,40/41/42 由 dspark_target 覆盖
    ratios = [0, 0] + [4 if (i % 2 == 0) else 128 for i in range(2, n)]
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": n,
        "num_hash_layers": 3,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 512,
        "index_share_for_mtp_iteration": True,
        "sliding_window": 128,
        "compress_ratios": ratios,
        "compress_rope_theta": 160000,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_block_size": 5,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 128799,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "topk_method": "noaux_tc",
        "routed_scaling_factor": 1.5,
        "max_position_embeddings": 1048576,
        "rope_theta": 10000,
        "rope_scaling": {"type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1},
        "vocab_size": 129280,
        "torch_dtype": "bfloat16",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-06,
    }


def _template_kimi_k3() -> Dict[str, Any]:
    n = 93
    full = [i for i in range(4, n, 4)] + [n]
    full_set = set(full)
    kda = [i for i in range(1, n) if i not in full_set]
    text = {
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"],
        "num_hidden_layers": n,
        "hidden_size": 7168,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "first_k_dense_replace": 1,
        "attn_res_block_size": 12,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "max_position_embeddings": 1048576,
        "intermediate_size": 33792,
        "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "moe_layer_freq": 1,
        "vocab_size": 163840,
        "rms_norm_eps": 1e-05,
        "hidden_act": "situ",
        "dtype": "bfloat16",
        "linear_attn_config": {
            "full_attn_layers": full,
            "kda_layers": kda,
            "head_dim": 128,
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    }
    return {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": text,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
    }


def _template_glm5() -> Dict[str, Any]:
    n = 45
    layer_types = [
        "deepseek_sparse_attention" if (i % 4 == 3) else "linear_attention"
        for i in range(n)
    ]
    full = [i for i in range(n) if layer_types[i] == "deepseek_sparse_attention"]
    full_set = set(full)
    kda = [i for i in range(n) if i not in full_set]
    text = {
        "model_type": "glm5_next_text",
        "num_hidden_layers": n,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 0,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "mla_use_nope": True,
        "first_k_dense_replace": 3,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (n - 3),
        "indexer_types": ["full"] * n,
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "indexer_rope_interleave": True,
        "index_share_for_mtp_iteration": True,
        "num_nextn_predict_layers": 1,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 288,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "n_group": 1,
        "topk_group": 1,
        "max_position_embeddings": 1048576,
        "vocab_size": 154880,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "dtype": "bfloat16",
        "linear_attn_config": {
            "full_attn_layers": full,
            "kda_layers": kda,
            "head_dim": 128,
            "num_heads": 64,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
    }
    return {"model_type": "glm5", "text_config": text, "tie_word_embeddings": False}


def template_source(model_key: str) -> Dict[str, Any]:
    """模板的 provenance 描述。"""
    repo = OFFICIAL[model_key][0]
    return {
        "kind": "template",
        "repo": repo,
        "url": None,
        "note": "报告 6.1/4.1 口径内置模板(离线兜底),建议联网抓官方 config 后重跑",
    }


def sha256_of_config(cfg: Dict[str, Any]) -> str:
    """对 config 的规范化 JSON 文本取 sha256,用于 provenance。"""
    text = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
