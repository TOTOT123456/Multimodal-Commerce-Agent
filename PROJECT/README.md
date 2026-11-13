# 100 工具电商导购 Agent

`PROJECT/` 包含有效的数据、配置、训练脚本、模型输出和实验报告。核心算法是 Intent 路由、工具检索、工具链规划三段式系统；多轮 Agent 在模型外增加状态解析与记忆，不改变这三段的 canonical 参数。

## 系统路线

```text
raw query
  → Qwen3-0.6B full LoRA Intent
  → BM25 + bge-large easy2 Embedding，各 Top-40
  → Weighted RRF：vector_weight=0.8, rrf_k=5
  → bge-reranker-v2-m3 exact Top-20
  → Source/Reranker RRF：source_weight=0.4, rrf_k=8
  → Top-10
  → Qwen3-8B cross-style SFT → RPO → DPO
  → ordered tool IDs
```

机器可读定义：

- `PROJECT/configs/retrieval_canonical_route.json`
- `PROJECT/configs/assignment3_current_dpo.json`
- `PROJECT/configs/agent_runtime.json`

## 数据与防泄漏

| 数据 | 条数 | 角色 |
|---|---:|---|
| `tool_call_trace/train.jsonl` | 24,000 | 训练侧唯一 Legacy 源 |
| `tool_call_trace/test.jsonl` | 6,000 | Legacy frozen 主评测 |
| `intent_recognition/intent_all.jsonl` | 9,000 | 6K 电商 + 3K 闲聊 frozen |
| API frozen | 1,000 | Planner 补充分布评测 |
| Planner cross-style SFT | 77,724 | 53,725 API-native + 23,999 Legacy |
| Planner DPO2 preference | 4,000 | 3,600 train + 400 monitor |
| Planner development | 1,200 | 锁定 checkpoint |
| Multi-turn dev/test | 991 / 993 | 各计划 1,000 段，语义审核排除 9/7 段；2～7 轮、距离 1～5、十二候选封存评测 |

数据生成后检查 normalized query、trace id、表达簇、短语和上下文骨架。冻结数据不用于训练、错误挖掘、偏好构造或选参。

## Intent

当前 Intent 端点为 Qwen3-0.6B `full_lora checkpoint-2000`，冻结 9K accuracy `1.0000`。它只输出路由类别；单轮链使用原始 query，多轮链使用确定性补全后的 `resolved_query`，均不消费生成关键词。

Qwen3-1.7B balanced LoRA 的 accuracy 为 `0.9996`，作为关键词能力对照保存。

## Retrieval

Embedding 以 `BAAI/bge-large-zh-v1.5` 为基座，用 MNRL 比较五类负例 recipe、三个随机种子，选择 `easy2/seed=20260731`。easy2 为每个 anchor 加入跨领域和新增无关显式负例。

一级检索将 BM25 和向量召回各 Top-40 用 Weighted RRF 融合。Reranker 为 `BAAI/bge-reranker-v2-m3`，负例 recipe `mixed_typed3_mined4`、学习率 `2e-5`，对一级前 20 全量打分。二级 RRF 在同一 20 候选域内输出 Top-10。

| Split | AllHit@10 | AllHit@20 |
|---|---:|---:|
| Legacy frozen 6K | 0.973667 | 0.995000 |
| Round-final 1.2K | 0.998333 | 1.000000 |
| API frozen 1K | 1.000000 | 1.000000 |

Reranker 深度 20、Planner 深度 10 来自联合质量/成本选择：比更深接口显著减少重排延迟和 Planner token，同时处于预设的近优质量带。

## Planner

### SFT

Qwen3-8B 使用 77,724 条跨风格样本训练 1 epoch：4×RTX 5090、LR `5e-5`、有效 batch 32、LoRA `r16/a32/d0.05`。端点：

`PROJECT/finetune/outputs/tooltrace-a3-crossstyle-sft8-4x5090-s20260728/checkpoint-2429`

### 偏好数据

从训练侧新 query 上挖掘当前父模型实际 greedy 错误；只有 strict 可解析、语义错误且 gold 完全位于 Top-10 的样本可以进入偏好池。长链 24K 面板得到 6,734 个错误，短链 12K 面板得到 1,683 个错误，跨面板去重后为 8,309 个。

最终 4,000 对覆盖全部 100 工具；工具数 1～6 为 `200/600/800/800/800/800`，主错误类型替换/遗漏/多选/纯顺序为 `2064/911/765/260`。每个 query/source 最多一对，最大长度 672，小于 768 上限。

### RPO 与 DPO

4 张 RTX 5090 并行完成四组主搜索：纯 DPO，以及 RPO alpha `0.5/1.0` 与两档学习率。主阶段选中 `RPO alpha=0.5, lr=1e-6, checkpoint-225`。因为开发最优位于边界，从它再训练 0.5 epoch，比较四个低学习率扩展配方。

28 个 checkpoint 在独立 1,200 条开发集上按 original-order RAG TraceEM 排序，最终端点：

`PROJECT/finetune/outputs/tooltrace-a3-planner-dpo2-ext-dpo-lr5e7-s20260819/checkpoint-113`

开发 TraceEM `0.888333`，相对起始父模型增加 `3.1667pp`，改善/回退 `42/4`，McNemar `p=5.10e-9`。

## 最终指标

| 评测 | SetEM | TraceEM | Strict | Conditional TraceEM |
|---|---:|---:|---:|---:|
| Legacy frozen 6K | 0.892000 | **0.889333** | 0.998833 | 0.913386 |
| API frozen 1K | 0.994000 | 0.994000 | 1.000000 | 0.994000 |

组合 0.6B Intent 后：

| Static system 9K | 正确数 | 比例 |
|---|---:|---:|
| Full Set | 8,352 | 0.928000 |
| Full Trace | 8,336 | **0.926222** |

## 多轮 Agent

多轮层采用“状态在模型外、历史按需读取、普通请求只向原 Planner 提交短 query”。只有指代、继承、元取消或修正才读取最近 2 个事件，明确远期引用再触发会话内精确 turn/BM25。订单取消、退款和购物车删除仍按当前业务请求处理，不会误删历史目标。SQLite 保存事件、紧凑状态和滚动摘要，并提供版本冲突检测、TTL 与按会话删除。

自然短会话评测的开发/独立测试各计划 1,000 段，语义审核后保留 991/993 段；每段 2～7 个用户轮，回放距离 1～5。计分目标覆盖自然确认、近期续办、对象线索和历史后的完整新请求，明确禁用“第 N 轮”“完整原样重做”等评测腔。`gpt-5.4-mini` 生成、独立 `gpt-5.5` 审核；相对 42,400 条保护数据及开发/测试之间的 query、来源 ID、表达簇、上下文骨架或完整会话重合均为 0。

| 候选 | 本步能力 | 开发混合 | 独立测试混合 | 测试回放 | 测试不继承 |
|---|---|---:|---:|---:|---:|
| A | 不读取历史 | 40.4% | 41.0% | 5.2% | 98.4% |
| B | 直接拼接最近两轮 | 51.9% | 50.5% | 59.2% | 36.5% |
| C | 结构化近期状态 | 65.9% | 62.9% | 48.0% | 86.9% |
| D | 加入规则摘要 | 76.4% | 74.1% | 66.2% | 86.9% |
| E | 加入按需记忆 | 87.4% | 85.5% | 84.6% | 86.9% |
| F | 加入近期路由优先级 | 87.4% | 85.5% | 84.6% | 86.9% |
| G | 加入请求边界 | 93.1% | 91.9% | 88.6% | 97.4% |
| H | 加入单工具目标归并 | 93.6% | 92.4% | 89.4% | 97.4% |
| I | 加入严格 Catalog 直达 | 93.6% | 92.4% | 89.4% | 97.4% |
| J | 加入完整意图恢复 | 96.6% | **94.6%** | **92.8%** | **97.4%** |
| K | 加入自包含保护 | 96.6% | **94.6%** | **92.8%** | **97.4%** |
| L | 加入同版本计划缓存 | 96.6% | **94.6%** | **92.8%** | **97.4%** |

开发集按“不继承最多比 A 低 1pp，再最大化回放”预注册规则锁定 L，之后才打开测试。L 在测试 993 个目标上正确 939 个；相对 A 净增 532 条、`+53.58pp`，相对 B 净增 438 条、`+44.11pp`。计划缓存使目标轮 Planner 调用由 993 次降至 394 次；但 L 的新请求点估计比 A 低 1.05pp（4 条净回退，配对 `p=0.125`），所以它是评测胜者，不是未经真实流量验证即可上线的结论。完整数据、十二候选、切片和误差见 [`reports/08_自然短会话记忆评测.md`](reports/08_自然短会话记忆评测.md)，工程架构见 [`reports/07_多轮对话与记忆系统设计.md`](reports/07_多轮对话与记忆系统设计.md)。

```bash
python3 -m unittest discover -s PROJECT/tests -v
python3 PROJECT/scripts/agent/audit_agent_static_regression.py
python3 PROJECT/scripts/agent/audit_agent_failure_guards.py
python3 PROJECT/scripts/agent/run_agent.py run --session demo --interactive
python3 PROJECT/scripts/agent/run_agent.py inspect --session demo
```

55 项 Agent 架构测试、故障/边界审计和真实 checkpoint 推理均已通过。Legacy 6K/API 1K 首轮 raw query 恒等率均为 1.0，严格快速路径触发均为 0，静态 TraceEM 仍为 0.889333/0.994000。多轮结果只代表 100 工具合成闭集，不等价于开放域对话或真实工具执行成功率。

## 目录

- `configs/`：catalog、数据和 canonical 路由。
- `data/tool_call_trace/`：Legacy train/frozen 与 Planner 实验产物。
- `data/intent_recognition/`：Intent 数据与统计。
- `data/rag/`：工具语料、Embedding/Reranker 数据和检索产物。
- `scripts/`：数据、训练、评测、选择和审计脚本。
- `agent/`：多轮状态、记忆、canonical 后端和编排器。
- `specs/`：Catalog、Agent 状态/事件和多轮评测数据契约。
- `tests/`：不加载模型的 Agent 架构回归测试。
- `finetune/`：训练缓存、输出、adapter 和预测。
- `reports/`：中文实验报告。

## 报告

- `reports/01_意图识别.md`
- `reports/02_检索增强.md`
- `reports/03_向量训练.md`
- `reports/04_数据集与训练用数说明.md`
- `reports/05_训练算法总览.md`
- `reports/06_Assignment3_100工具实验总报告.md`
- `reports/07_Planner再训练_20260808.md`
- `reports/07_多轮对话与记忆系统设计.md`
- `reports/08_自然短会话记忆评测.md`

## 核心复现入口

从仓库根目录执行：

```bash
python3 PROJECT/scripts/catalog/validate_tool_catalog.py
python3 PROJECT/scripts/dataset/audit_generation_leakage.py --tokenizer fallback
python3 PROJECT/scripts/dataset/audit_datasets.py --strict-min-test-per-tool 30
```

Planner DPO2 的脚本入口：

```bash
bash PROJECT/scripts/finetune/run_assignment3_planner_dpo2_sweep.sh
bash PROJECT/scripts/finetune/run_assignment3_planner_dpo2_extension.sh
bash PROJECT/scripts/finetune/run_assignment3_planner_dpo2_dev_eval.sh
```

训练和预测产物体积较大，`PROJECT/models/`、本地环境与临时缓存不应提交。

## 结果边界

当前结果只覆盖 100 工具合成闭集。Full Trace 不等价于真实工具执行成功率；新工具、真实流量、参数生成、真实 API 故障和生产多租户记忆仍需单独评测。多轮独立测试计划 1,000 段、审核保留 993 段受控合成短会话，覆盖完整历史任务回放与完整新请求切换，最远跨 5 轮；来源与目标是模型生成的自然语言，不代表独立真实流量。测试集一次性运行预注册的十二候选，后续不得用它调规则。
