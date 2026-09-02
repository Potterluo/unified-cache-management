# ucm/sandbox/fake_model —— 按官方注意力类型与层分布生成"假模型"

在**单卡 NPU(如 64GB)** 上验证 UCM 缓存架构(规格表 + 链式/快照双原语)用的实验性
工具集:只保留目标模型**官方 config.json 的层分布与注意力/KV 形状**,把层数砍到
N 层(默认 8),配合 vllm 的 `--load-format dummy` 随机初始化权重,不下载任何真实
权重文件——目标模型真实权重单卡装不下:如 DeepSeek-V4-Flash-0731 约 **240-250B 参数
(社区口径),Q8 量化约 162GB**,单张 64GB NPU 放不下;UCM 验证只需要"真实架构的
KV 组结构"——FULL/MLA/CSA/SWA/KDA/DSA/Indexer 的分组与形状,因此**减层后 KV 相关
形状(hidden/heads/head_dim/lora ranks/sliding_window/compress_ratios/indexer 等)
必须原样保留**,才能让 vllm-ascend 的 KVCacheConfig 分组与真实模型一致。

对应《UCM_缓存设计分析报告_2026-08.md》第 6.1 节("模型波次 → 缓存种类")与
4.1 节(规格表)的口径。**本工具只产出代码 + 配置/计划文件,不连接服务器/NPU。**

---

## 目录结构

```
ucm/sandbox/fake_model/
├── build_fake_model.py      # 主 CLI:抓 config → 解析逐层注意力 → 缩减 config + layer_plan
├── layer_plans.py           # 纯逻辑:逐层分类 / KV 组推导 / 缩减规则 / 报告口径模板
├── fake_weights.py          # 可选:离线随机权重 safetensors 生成(纯 stdlib,不需要 torch)
├── test_fake_model.py       # unittest(不依赖 vllm/torch/ucm 包)
├── official_configs/        # 已抓取缓存的官方 config.json(见下表)
│   ├── deepseek-v4.json     #   deepseek-ai/DeepSeek-V4-Flash-0731
│   ├── kimi-k3.json         #   moonshotai/Kimi-K3
│   └── glm-5.3.json         #   zai-org/GLM-5.3-Flash
└── README.md
```

> **无污染说明**:`ucm/__init__.py` 会 `from ucm.integration.vllm.patch.logger_patch import
> patch_logger`(依赖 `wrapt` 等)。因此本工具集**刻意不 import `ucm` 包、不设
> `__init__.py`**,三个 `.py` 模块互相之间也只是同级模块导入——`python3` 标准库即可跑,
> `requests`/`torch`/`numpy` 全是可选加速;缺了都能优雅降级。

---

## 使用方式 (a):仅 config + `--load-format dummy`(主路径,推荐)

```bash
# 研究模式:只打印 layer_plan(不写文件)
python build_fake_model.py --model deepseek-v4 --layers 8

# 生成缩减模型目录
python build_fake_model.py --model deepseek-v4 --layers 8 --out ./fake_dsv4
#   -> ./fake_dsv4/layer_plan.json       逐层类型 + KV 组(给 UCM 规格表用)
#   -> ./fake_dsv4/config.json           缩减版 config(喂给 vllm)
#   -> ./fake_dsv4/official_config.json  官方原始 config(备查)

# 单卡 NPU 上用 vllm-ascend 跑(无需真实权重文件)
vllm serve ./fake_dsv4/config.json \
    --load-format dummy \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 ...
```

关键点:缩减版 `config.json` **只砍层数、只缩小 MoE/FFN 字段**,而
`hidden_size / num_attention_heads / num_key_value_heads / head_dim / kv_lora_rank /
q_lora_rank / qk_rope_head_dim / qk_nope_head_dim / v_head_dim / sliding_window /
compress_ratios(前 N 项) / indexer 参数 / mamba 形状` 等 **KV 相关形状字段一字不动**,
保证 vllm 的 KVCacheConfig 分组结构与真实模型一致。`num_experts / intermediate_size /
moe_intermediate_size` 默认缩小若干倍(可用 `--keep-ffn` 取消),以降低 dummy 权重
显存——注意这**不改变 KV 分组结构**。

### 官方 config 获取优先级

1. `official_configs/<model>.json`(本仓库已缓存,离线可用);
2. `--fetch`:从 HuggingFace 官方仓库抓取并回写缓存;
3. 全部失败 → 内置"报告 6.1 口径"模板(`layer_plans.py` 的 `template_config`)。

官方来源(2026-08 抓取):

| model_key | 官方仓库 | config URL | 缓存文件 | 字节 |
|---|---|---|---|---|
| deepseek-v4 | deepseek-ai/DeepSeek-V4-Flash-0731 | <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/main/config.json> | deepseek-v4.json | 1 888 |
| kimi-k3 | moonshotai/Kimi-K3 | <https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json> | kimi-k3.json | 7 006 |
| glm-5.3 | zai-org/GLM-5.3-Flash | <https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json> | glm-5.3.json | 69 416 |

`layer_plan.json` 的 `source` 字段记录了所用来源(kind=official_config/template、URL、
本地文件、sha256)。

## 使用方式 (b):离线随机权重 safetensors(可选)

```bash
# 由主 CLI 顺带生成(--out 必需)
python build_fake_model.py --model kimi-k3 --layers 8 --out ./fake_k3 --weights --shrink-vocab 4096

# 或独立使用
python fake_weights.py --plan ./fake_k3/layer_plan.json --config ./fake_k3/config.json \
    --out ./fake_k3/weights --seed 1234 --shards 1
```

`fake_weights.py` 是**纯 stdlib 实现的随机权重生成器**(不需要 torch/numpy;有 numpy
时自动提速)。重要限制:权重名/形状是按常见 HF 命名约定 + layer_plan 的注意力种类
**启发式**生成的,**不保证**与 vllm 各模型实现(DeepseekV4/KimiLinear/GLM5)的显式
参数名一一对应——**vllm 推理请始终走 `--load-format dummy`**;这套 saftetensors 是给
UCM 自己的验证代码(如双原语/形状解析)准备"像真的文件"。另外注意官方 head_dim
很大(如 DSV4 是 64 头 × 512),生成的 q/o 投影张量天然巨大(这是 KV 形状不能缩的
代价),可用 `--shrink-vocab` 削减词表两个大张量。

---

## 层分布 → KV 组种类映射表(官方 config 解析结果)

> 类型串记号:full=全注意力、mla=MLA 压缩 KV、c4/c128=CSA C4/C128、kda=KDA 线性
> 注意力、dsa=DeepSeek 稀疏注意力、dspark=DeepSeek Spark 哈希稀疏层。
> 组种类:chain=链式(积木)、snapshot=快照(罐头)、sidecar=侧车(索引器,不投票);
> block=链式块大小 / 快照网格粒度。

### DeepSeek-V4-Flash-0731(`deepseek-v4`,43 层 + 3 hash 层)

| 层区间 | 注意力类型 | KV 组(种类, block) | 依据(config 字段) |
|---|---|---|---|
| 0–1 | full(全注意力,MLA 形态) | `full`(chain, 128) | `compress_ratios[0..1]=0` |
| 2–39 偶 | csa C4 | `csa_c4`(chain, 128, storage=128/4=32) | `compress_ratios=4` |
| 2–39 奇 | csa C128 | `csa_c128`(chain, 128, storage=128/128=1) | `compress_ratios=128` |
| 0–42 | SWA 每层分支 | `swa`(chain, 128, window=128) | `sliding_window=128`(6.1"SWA 每层分支") |
| 2–42(CSA 源) | 索引器(侧车) | `indexer`(sidecar, topk=512) | `index_topk=512, index_n_heads=64, index_head_dim=128` |
| 40–42 | dspark 哈希稀疏 | `dspark`(snapshot, block=5) | `dspark_target_layer_ids=[40,41,42], dspark_block_size=5` |

前 8 层类型串:`full,full,c4,c128,c4,c128,c4,c128`
KV 形状:`hidden=4096, heads=64, head_dim=512, q_lora_rank=1024, o_lora_rank=1024,
qk_rope_head_dim=64, kv_heads=1, sliding_window=128`

### Kimi K3(`kimi-k3`,93 层)

| 层区间 | 注意力类型 | KV 组(种类, block) | 依据(config 字段) |
|---|---|---|---|
| 0(首层 dense)∪ {4,8,…,92} | mla | `mla`(chain, 128) | `first_k_dense_replace=1` + `linear_attn_config.full_attn_layers` |
| 其余 69 层 | kda 线性注意力 | `kda`(snapshot, 逐 token block=1) | `linear_attn_config.kda_layers` |

MLA 与 KDA **共用同一个分页块池**(6.1"状态与 MLA KV 共用同一个分页块池"),
两组都标注 `shared_pool: k3_mixed_pool`。全层 MoE(`moe_layer_freq=1`)。
前 8 层类型串:`mla,kda,kda,kda,mla,kda,kda,kda`
KV 形状:`hidden=7168, heads=96, head_dim=128, kv_lora_rank=512, q_lora_rank=1536,
qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128`

### GLM-5.3-Flash(`glm-5.3`,45 层)

| 层区间 | 注意力类型 | KV 组(种类, block) | 依据(config 字段) |
|---|---|---|---|
| 34 层 KDA(3 连一组的 linear_attention) | kda 线性注意力 | `kda`(snapshot, block=1) | `layer_types` / `linear_attn_config.kda_layers`(34 个) |
| 11 层 dsa(每第 4 层) | DeepSeek 稀疏注意力 | `dsa`(chain, 128) | `layer_types` / `full_attn_layers`(11 个) |
| 11 层 dsa(索引器) | 索引器(侧车) | `indexer`(sidecar, topk=2048, kpool=4) | `index_topk=2048, index_kpool=4`(6.1"索引器 top-k 2048") |

前 3 层 MLP 为 dense,其余 42 层 MoE(`mlp_layer_types`)。前 8 层类型串:
`kda,kda,kda,dsa,kda,kda,kda,dsa`
KV 形状:`hidden=4096, heads=64, qk_head_dim=256(rope=0), v_head_dim=256,
kv_lora_rank=512, q_lora_rank=1536;linear_attn_config.head_dim=128`

---

## layer_plan.json 结构(片段)

```jsonc
{
  "schema_version": 1,
  "model_key": "deepseek-v4",
  "official_repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
  "source": { "kind": "official_config", "url": "…", "local_file": "…", "sha256": "…" },
  "num_hidden_layers_original": 43,
  "num_layers_requested": 8,
  "num_layers": 8,
  "cfg_kv_shape": { /* KV 关键形状标量快照(缩减前后必须一致) */ },
  "layer_plan": [
    { "index": 0, "type": "full", "group": "full", "mlp": "moe",
      "sliding_window": 128, "params": { "head_dim": 512, "q_lora_rank": 1024, … } },
    { "index": 2, "type": "csa", "group": "csa_c4", "mlp": "moe",
      "sliding_window": 128, "params": { "compress_ratio": 4, "storage_block_size": 32 } },
    …
  ],
  "kv_groups": [
    { "name": "full",   "kind": "chain",    "block_size": 128, "layers": [0, 1], "seed": "S_full", … },
    { "name": "csa_c4", "kind": "chain",    "block_size": 128, "layers": [2, 4, 6], "seed": "S_csa_c4",
      "params": { "compress_ratio": 4, "storage_block_size": 32 }, "per_token_bytes": …, "estimate": true },
    { "name": "swa",    "kind": "chain",    "block_size": 128, "layers": [0…7],
      "params": { "sliding_window": 128, "per_layer_branch": true }, "seed": "S_swa" },
    { "name": "indexer","kind": "sidecar",  "block_size": 0,   "layers": [2, 3, …],
      "params": { "index_topk": 512, … }, "seed": "S_indexer" },
    …
  ],
  "type_string": "full,full,c4,c128,c4,c128,c4,c128"
}
```

种子 `S_*` 沿用报告 4.1 的哈希隔离命名;`per_token_bytes` 为量级估算(`estimate: true`,
ML 缓存压缩形态下仅供参考,见报告 4.1"每层字节 = 示例值")。

---

## 测试

```bash
# 在 ucm/sandbox/fake_model/ 目录下(任务指定命令;pytest ≤ 8.3 默认 import 模式)
python -m pytest test_fake_model.py -c <repo>/test/pytest.ini --confcutdir=<repo>/test/suites/Unit
# 或从仓库根目录
python -m pytest ucm/sandbox/fake_model/test_fake_model.py -c test/pytest.ini \
    --confcutdir=test/suites/Unit
```

`--confcutdir` 保证不加载 `test/conftest.py`(它拖 pynvml 等依赖)。测试覆盖:
(a) 缩减 config 层数 == N;(b) 前 N 层层类型模式与原始/模板逐层一致;
(c) KV 形状字段(含 compress_ratios 前缀)保留;(d) layer_plan.json 结构合法
(必填键/逐层序号/组唯一/种子/来源);(e) 报告模板离线兜底;(f) 研究模式只打印
JSON;(g) safetensors 纯 stdlib 写读回。本机实测:Python 3.11.9 + pytest 7.4.4,
**21 passed**。

> pytest ≥ 9.0 注意:新版 pytest 按 rootdir 相对路径推导模块名,而任务命令的
> `-c <repo>/test/pytest.ini` 使 rootdir=<repo>/test;由于测试文件在 `<repo>/ucm/` 下、
> `ucm/__init__.py` 是常规包(会 import wrapt 等),pytest 9 会把整个祖先包链引入。
> 这是在 pytest 8/7(默认 prepend 导入)上验证通过的;若环境是 pytest ≥ 9,建议
> 用系统 Python 的 pytest ≤ 8.3 跑本套测试,或给 `ucm/__init__.py` 补齐其依赖。

---

## 把缩减 config 喂给 vllm-ascend `--load-format dummy` 时最可能卡的点

1. **model_type 必须被引擎注册**:`deepseek_v4` / `kimi_k3` / `glm5_next_text` 需要
   vllm(vllm-ascend)的模型类支持。`--load-format dummy` 只绕开权重文件**,不绕开
   模型实现注册**——K3/GLM 属新架构,大概率依赖本仓库 `ucm/integration/vllm/patch/`
   的补丁(如 DS-V4 已有 v0270 补丁);先确认 `vllm/model_executor/models/__init__.py`
   里模型名已挂上。
2. **层分布字段的长度一致性**:vllm 的 `DeepseekV4` 等实现可能断言
   `len(compress_ratios) == num_hidden_layers(+num_hash_layers)`。缩减后务必让
   `compress_ratios` 截到前 N 项、`dspark_target_layer_ids` 过滤到 `<N`(工具已做);
   若 N ≥ 43 则保留官方原样(注意官方 config 的 46 项数组本身就比 43 长,引擎必须
   容忍)。K3/GLM 的 `full_attn_layers`/`kda_layers` 同样只保留 `<N` 的层号。
3. **KV 形状字段是硬依赖**:`head_dim / num_attention_heads / num_key_value_heads /
   q_lora_rank / o_lora_rank / kv_lora_rank / qk_rope_head_dim / qk_nope_head_dim /
   v_head_dim / sliding_window` 这些一旦被 vllm 读取并参与 KVCacheConfig 计算,任何
   改动都会让分组结构与真实模型不一致——这正是本工具"KV 形状必须原样"的原因。
   DSV4 的 MLA 层 config 里**没有 `kv_lora_rank`**(只有 q/o lora rank),要靠引擎
   推断压缩 KV 形态;K3 的 MLA 有 `kv_lora_rank`。GLM 顶层 `head_dim=0`(真实维在
   `linear_attn_config.head_dim`),若引擎从顶层断言 `head_dim>0` 会先炸。
4. **vllm-ascend 分组与对齐约束**:`storage_block_size = block_size // compress_ratio`
   (c4→32、c128→1)、`kernel_block_size=128` 硬对齐、DS-V4 双 FA 组(C4/C128)**各自
   截断后取公倍数**。缩减层数后如果某个组(如 dspark 或 kda)不存在(N 太小),要确
   认引擎不要求"至少一个全注意力组"或"组数 ≥2"(vllm ≤0.11 曾硬断言恰好两种组,
   0.18+ 放开;vllm-ascend 0.27/0.28 见补丁)。
5. **索引器/侧车字段**:DSV4 的 `index_topk/index_n_heads/index_head_dim`、GLM 的
   `index_topk=2048/index_kpool=4` 若引擎的 indexer(如 `DeepseekV4IndexerCache`,
   `tokens_per_state=compress_ratio`)不支持或缺失会报配置错误;这些字段本工具原样
   保留,引擎侧需要对应实现。
6. **多模态外设**:K3/GLM 都带 `vision_config`(vllm 会连 vision tower 一起
   dummy 加载,占显存)。如只想验证 KV 分组,可 `--drop-vision`,但需确认引擎不要求
   图像处理器;稳妥做法是保留。
7. **词表/嵌入**:`vocab_size` 与 `tie_word_embeddings` 决定 embedding/lm_head 两个
   大张量(占 dummy 权重显存的相当一部分);`--shrink-vocab N` 可降,但 vllm 可能
   校验 tokenizer 词表与 config 一致(离线 dummy 场景一般只警告)。
8. **MoE 字段一致性**:缩小 `n_routed_experts / num_experts_per_tok` 时保证
   `per_tok ≤ routed`(工具已钳制),`topk_group/n_group/routed_expert_hidden_size`
   等成组字段别自相矛盾;`num_nextn_predict_layers`(DSV4/GLM 都有 MTP)保持官方值。

---

## 边界与免责

- 2026 年这批"报告口径"模型为官方已发布模型,config 随官方更新;若官方 config 变化,
  重新 `--fetch` 后缓存与 sha256 会随之更新,`layer_plan` 以真 config 为准。
- `fake_weights.py` 是启发式参数清单,**不能**替代 `--load-format dummy` 的权重;
  只服务于 UCM 侧的形状/原语验证。
- 本工具不发起任何 NPU/服务器连接;唯一的网络行为是可选的官方 config 抓取。