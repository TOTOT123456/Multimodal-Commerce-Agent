# 多模态智能购物对话意图识别大模型系统

面向电商客服场景的图文对话理解项目，聚焦 **图片场景分类** 与 **多轮对话意图识别**。围绕数据构建、Caption / VQA、两阶段 SFT 和 GRPO 组织训练方案，并提供电商 Agent 的意图路由、工具检索、调用顺序规划与上下文管理代码。

## 任务说明

- **图片场景分类**：识别商品图片、订单页面、物流页面及售后截图等电商场景。
- **对话意图识别**：结合用户问题、历史对话与图片线索，判断商品材质、商品规格、售后申请、使用方式等需求。
- **Agent 工具规划**：将用户需求映射到业务工具，从 100 个电商工具中检索候选并生成有序工具 ID。

## 多模态训练方案

以 **Qwen2.5-VL-7B-Instruct** 为基座，训练路线分为数据工程、监督微调与强化学习三个环节。

### 数据工程

1. 检查原始标签冲突和类别混淆，通过多模型投票与人工抽样复核改善标签一致性。
2. 结合 OCR 文本与商品图片组织 Caption 数据，覆盖页面结构、商品属性和关键文字信息。
3. 将图片、用户问题及对话历史组织为 VQA 样本，补充多轮表达和长尾类别。
4. 按数据来源与任务划分训练、验证和测试集合，并检查样本重叠。

### 两阶段 SFT

- **Stage 1：领域适配。** 使用电商 Caption 数据进行全参微调，强化电商图像与页面内容理解。
- **Stage 2：任务对齐。** 使用 VQA 数据进行 LoRA 微调，围绕场景分类和意图识别对齐输出；方案中的 LoRA rank 为 16。

### GRPO 方案

通过思考链样本冷启动建立输出格式，再结合答案正确性、格式、推理与回答一致性及置信度设计奖励。关注重复输出、多标签投机与过长推理等 reward hacking 行为，并通过独立分类评估与 bad case 分析检查优化效果。

## Agent 功能与代码导航

| 模块 | 功能 | 代码入口 |
| --- | --- | --- |
| 意图路由 | 区分电商请求与闲聊，决定是否进入工具规划 | `PROJECT/agent/backends.py` |
| 混合检索 | BM25 与向量召回、RRF 融合、Reranker 精排 | `PROJECT/configs/retrieval_canonical_route.json` |
| 工具规划 | 根据候选工具生成有序工具 ID，并校验结构与范围 | `PROJECT/agent/orchestrator.py` |
| 多轮上下文 | 处理指代、确认、取消、修正与历史任务回放 | `PROJECT/agent/resolver.py` |
| 会话记忆 | 使用 SQLite 保存事件、状态和摘要 | `PROJECT/agent/memory.py` |
| 运行控制 | Token 预算、会话隔离、版本冲突与 TTL | `PROJECT/agent/types.py`、`memory.py` |
| 离线验证 | 架构测试、多轮评测、数据审计与回归检查 | `PROJECT/tests/`、`PROJECT/scripts/agent/` |

```text
用户问题与会话历史
        │
        ▼
上下文整理 → 意图路由
        │
        ▼
BM25 + Embedding → RRF → Reranker
        │
        ▼
Planner → 有序工具 ID → 结构校验
        │
        ▼
保存事件、会话状态与摘要
```

Agent 配置使用 Qwen3 Intent / Planner 与 BGE Embedding / Reranker；相应文本训练脚本采用 SFT 和 DPO / RPO。

## 技术栈

| 方向 | 技术 |
| --- | --- |
| 多模态训练方案 | Qwen2.5-VL、Caption / VQA、OCR、SFT、LoRA、GRPO、DeepSpeed、vLLM |
| Agent 与检索代码 | Python、SQLite、PyTorch、Transformers、MS-SWIFT、Sentence-Transformers、BM25、RRF、Reranker、DPO / RPO |

## 快速开始

建议使用 Python 3.10 或更新版本，在仓库根目录运行无模型架构测试：

```bash
python3 -B -m unittest PROJECT.tests.agent.test_agent_runtime
python3 -B PROJECT/scripts/catalog/validate_tool_catalog.py
```

配置好模型路径、LoRA adapter、Embedding / Reranker 与检索索引后，启动 Agent：

```bash
python3 PROJECT/scripts/agent/run_agent.py run --session demo --interactive
python3 PROJECT/scripts/agent/run_agent.py inspect --session demo
```

模型及检索配置见 `PROJECT/configs/`。架构测试使用模拟后端，实际模型推理需准备对应依赖与模型产物。

## 目录结构

```text
PROJECT/
├── agent/        # 编排、上下文、状态与记忆
├── configs/      # 工具目录、模型、检索与评测配置
├── data/         # 工具轨迹、意图数据与工具语料
├── scripts/      # 文本数据构建、训练、检索、评测与审计
├── specs/        # 工具、状态、事件和评测数据契约
├── tests/        # Agent 架构测试
└── reports/      # 文本实验报告与实现说明
```

更多 Agent 实现细节与实验说明见 [PROJECT/README.md](PROJECT/README.md)。
