# Assignment 3：100 工具电商导购 Agent 实验总报告

## 1. 最终系统

当前系统由三个固定模块组成：

```text
原始 query
  → Qwen3-0.6B Intent LoRA
  → BM25 + easy2 Embedding，各 Top-40
  → Weighted RRF(vector_weight=0.8, k=5)
  → bge-reranker-v2-m3 exact Top-20
  → 二级 RRF(source_weight=0.4, k=8)
  → Top-10
  → Qwen3-8B SFT → RPO → DPO Planner
  → 有序工具 ID Trace
```

唯一机器可读配置是 `PROJECT/configs/retrieval_canonical_route.json`。Planner 端点是：

`PROJECT/finetune/outputs/tooltrace-a3-planner-dpo2-ext-dpo-lr5e7-s20260819/checkpoint-113`

## 2. 任务与指标

系统在 100 个已知电商工具组成的闭集上完成：

1. Intent：区分电商请求与闲聊；
2. Retrieval：确保 gold 工具均进入 Planner 可见 Top-10；
3. Planner：从 Top-10 中选择完整且有序的工具 ID Trace。

主要指标：

- `AllHit@10`：所有 gold 工具都进入 Top-10；
- `SetEM`：预测工具集合与 gold 完全一致；
- `TraceEM`：工具集合和顺序都完全一致；
- `Strict`：输出严格符合约定格式；
- `Full Trace`：Intent 路由正确，且电商样本的有序工具链正确。

Full Trace 是静态准确率，不包含工具参数生成与真实执行。

## 3. 数据和冻结边界

| 数据 | 数量 | 用途 |
|---|---:|---|
| 工具 catalog | 100 | 检索与候选定义 |
| Legacy train | 24,000 | 训练侧生成、SFT 与错误挖掘来源 |
| Legacy frozen test | 6,000 | 锁定后主评测 |
| Intent frozen | 9,000 | 6,000 电商 + 3,000 闲聊 |
| API frozen | 1,000 | Planner 补充分布评测 |
| Planner SFT | 77,724 | 53,725 API-native + 23,999 Legacy |
| Planner DPO2 preference | 4,000 | 3,600 train + 400 monitor |
| Planner development | 1,200 | checkpoint 选择 |

数据构建后重新审计 query、trace id、表达簇和上下文骨架；训练侧与冻结侧重叠均为 0。Legacy 6K、API 1K 和 Intent 9K 不用于偏好构造或 Planner 选点。

## 4. Intent

Intent 使用 Qwen3-0.6B full LoRA `checkpoint-2000`。冻结 9K accuracy 为 `1.0000`。检索直接消费原始 query，不消费 Intent 模型输出的关键词，因此分类输出和检索 query 之间不存在隐式改写。

Qwen3-1.7B balanced LoRA 的冻结 accuracy 为 `0.9996`，保留为关键词能力对照，不进入当前执行链。

## 5. 检索

### 5.1 Embedding

Embedding 从 `BAAI/bge-large-zh-v1.5` 出发，用 Multiple Negatives Ranking Loss 训练。当前 easy2 recipe 同时加入跨领域显式负例和新增无关显式负例，seed 为 `20260731`。

### 5.2 一级召回

BM25 与向量召回各取 40 个候选，通过 Weighted RRF 融合：

$$
S(d)=\frac{1-0.8}{5+r_{\mathrm{BM25}}(d)}+\frac{0.8}{5+r_{\mathrm{vec}}(d)}
$$

### 5.3 Reranker 与二级融合

`bge-reranker-v2-m3` 使用 `mixed_typed3_mined4` 负例配方、学习率 `2e-5`，对一级前 20 个候选逐一打分。随后在同一 20 候选域中，将一级排序和 reranker 排序按 `source_weight=0.4, k=8` 再做 RRF，输出 Top-10。

Legacy frozen 6K 的 `AllHit@10=0.973667`；API frozen 1K 为 `1.000000`。因此 Legacy 端到端 Planner 的理论检索上限是 97.3667%。

## 6. Planner SFT

Qwen3-8B 使用 77,724 条跨风格数据完成 1 epoch LoRA SFT。训练使用 4×RTX 5090、学习率 `5e-5`、有效 batch 32、LoRA `r16/a32/d0.05`。端点 `checkpoint-2429` 的结果为：

| 评测 | SetEM | TraceEM | Strict |
|---|---:|---:|---:|
| Legacy 6K | 86.1333% | 85.6833% | 99.7833% |
| API 1K | — | 99.5000% | 100.0000% |

组合 0.6B Intent 后，9K Full Trace 为 `90.4556%`。SFT 已完成主要分布与格式学习，后续错误以具体工具替换、漏选、多选和顺序问题为主，适合使用偏好优化。

## 7. Planner 偏好优化

### 7.1 数据构造

使用起始父模型在训练侧新 query 上的真实 greedy 输出做 rejected，只保留 strict 可解析且语义错误的样本。长链 24K 面板产生 6,734 个错误，短链 12K 面板产生 1,683 个错误，跨面板去重池为 8,309。最终按工具数和错误类型分层选择 4,000 对：

| 工具数 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|
| 对数 | 200 | 600 | 800 | 800 | 800 | 800 |

| 错误 | 替换 | 遗漏 | 多选 | 纯顺序 |
|---|---:|---:|---:|---:|
| 对数 | 2,064 | 911 | 765 | 260 |

每个 query/source 只保留一对，覆盖全部 100 工具；最大长度 672，训练上限 768，无截断。

### 7.2 训练策略

4 张 5090 并行比较纯 DPO 与 RPO chosen-NLL 锚点。主阶段最优配方是 `DPO + RPO, alpha=0.5, lr=1e-6`。因为最优 checkpoint 位于 1 epoch 边界，从它自适应延长 0.5 epoch，并比较四个低学习率配方；reference adapter 始终固定。

总计评测 28 个 checkpoint，按独立开发集 original-order RAG TraceEM 选中纯 DPO `lr=5e-7, checkpoint-113`。冻结集未参与选择。

### 7.3 开发集证据

1,200 条独立开发集 `AllHit@10=0.966667`。最终 Planner TraceEM 为 `0.888333`，相对起始父模型 `0.856667` 提升 `3.1667pp`；逐条改善/回退 `42/4`，exact McNemar `p=5.10e-9`。

## 8. 最终结果

### 8.1 Legacy frozen 6K

| 指标 | 数值 |
|---|---:|
| AllHit@10 | 97.3667% |
| SetEM | 89.2000% |
| TraceEM | **88.9333%** |
| Strict | 99.8833% |
| Conditional TraceEM | 91.3386% |

相对 SFT，TraceEM 提升 `3.2500pp`；相对起始 DPO，提升 `3.1333pp`。与起始 DPO 的逐条改善/回退为 `222/34`。

### 8.2 API frozen 1K

`SetEM/TraceEM/Strict=0.994/0.994/1.000`。与起始 DPO 相比为 `1/2` 改善/回退，净变化 `-0.1pp`。

### 8.3 静态系统 9K

| 指标 | 正确数 | 比例 |
|---|---:|---:|
| Full Set | 8,352 / 9,000 | 92.8000% |
| Full Trace | 8,336 / 9,000 | **92.6222%** |

系统 Full Trace 相对 SFT 增加 `2.1667pp`，相对起始 DPO 增加 `2.0889pp`。

## 9. 误差解释

Legacy 6K 的检索上限为 97.3667%，最终 TraceEM 为 88.9333%。在检索成功子集上，条件 TraceEM 为 91.3386%，说明剩余误差同时来自：

- 158 条左右的检索不可达样本；
- Planner 在高相似工具之间的替换和漏选；
- 长链中的局部顺序错误；
- 极少量输出格式错误。

下一轮若继续优化，应先在全新训练侧分布确认哪一类仍可稳定挖出，再决定继续 DPO、扩充 SFT，还是提高检索上限；冻结集不能用于构造下一轮偏好。

## 10. 证据入口

- canonical 路由：`PROJECT/configs/retrieval_canonical_route.json`
- DPO 配置：`PROJECT/configs/assignment3_current_dpo.json`
- 偏好审计：`PROJECT/data/tool_call_trace/_aux/assignment3/assignment3_planner_dpo2_20260808/preferences/preference_audit.json`
- checkpoint 选择：`PROJECT/data/tool_call_trace/_aux/assignment3/assignment3_planner_dpo2_20260808/selection/dpo_checkpoint_selection.json`
- 最终汇总：`PROJECT/data/tool_call_trace/_aux/assignment3/assignment3_planner_dpo2_20260808/final_summary.json`
- Planner 详报：`PROJECT/reports/07_Planner再训练_20260808.md`

## 11. 适用范围

结论只适用于 100 工具合成闭集和固定 Top-10 接口。当前实验未评测新增工具冷启动、真实流量表达漂移、工具参数生成、真实 API 执行、交易状态变化或多轮失败恢复。
