# 多模态智能购物对话意图识别大模型系统

面向电商客服场景的意图识别与工具规划项目，用于理解用户需求、检索业务工具、规划调用顺序，以及在多轮对话中维护必要上下文。

**当前代码包含文本电商 Agent 基线；尚未包含图像场景分类、Qwen2.5-VL 微调和 GRPO 训练实现。** 本仓库使用多模态应用名称组织扩展方向，现有功能与扩展设计分别说明。

## 当前实现

| 模块 | 功能 | 代码入口 |
| --- | --- | --- |
| 意图路由 | 区分电商请求与闲聊，决定是否进入工具规划 | `PROJECT/agent/backends.py` |
| 工具检索 | BM25 与向量召回、RRF 融合、Reranker 精排 | `PROJECT/configs/retrieval_canonical_route.json` |
| 工具规划 | 根据候选工具输出有序工具 ID，校验结构与范围 | `PROJECT/agent/orchestrator.py` |
| 多轮上下文 | 指代、确认、取消、修正与历史任务回放 | `PROJECT/agent/resolver.py` |
| 会话记忆 | SQLite 持久化事件、状态、摘要和会话内记忆 | `PROJECT/agent/memory.py` |
| 预算与边界 | 输入 Token 预算、会话隔离、版本冲突、TTL 和删除 | `PROJECT/agent/types.py`、`memory.py` |
| 离线验证 | 架构测试、回归审计、数据隔离和多轮评测 | `PROJECT/tests/`、`PROJECT/scripts/agent/` |

工具目录包含 **100 个已知电商工具**。当前输出为规划结果，不等同于真实工具参数生成、业务 API 执行或交易完成。

## 文本基线架构

```text
当前输入 + 会话标识
        │
        ▼
上下文路由与必要历史补全
        │
        ▼
Intent 路由：电商 / 闲聊
        │
        ▼
BM25 + Embedding → RRF → Reranker → Top-10
        │
        ▼
Planner → 结构校验 → 有序工具 ID
        │
        ▼
保存事件、状态与摘要
```

现有配置包括 Qwen3-0.6B Intent、BGE Embedding / Reranker 与 Qwen3-8B Planner。文本 Planner 使用 SFT 和 DPO / RPO；这与多模态设计中的 Qwen2.5-VL、GRPO 是不同实验路线。

## 多模态扩展设计

目标是结合用户图片、当前问题与历史对话，完成图片场景分类和对话意图识别，再将结果接入工具规划。

计划探索：

1. 标注冲突检查、多模型伪标签与人工抽样复核。
2. Caption / VQA 数据构建，以及训练、验证、测试的来源隔离。
3. 基于 Qwen2.5-VL 的领域适配 SFT 与 LoRA 任务对齐。
4. 在监督基线与独立评测建立后，研究冷启动和 GRPO 奖励设计。
5. 分别评估正确性、类别覆盖、无效输出、置信度与推理成本。

**本节是设计方向。当前代码未提供对应的多模态训练脚本、图像数据、权重或可核验的多模态 F1 结果，不将方案描述作为已完成实验。**

## 技术栈

- 当前实现：Python、SQLite、PyTorch、Transformers、MS-SWIFT、Sentence-Transformers、BM25、Embedding、Reranker、LoRA、DPO / RPO。
- 多模态扩展：Qwen2.5-VL、Caption / VQA、OCR、DeepSpeed、vLLM、GRPO；对应代码有待补齐。

## 本地验证

使用 Python 3.10 或更新版本，从仓库根目录执行：

```bash
python3 -B -m unittest PROJECT.tests.agent.test_agent_runtime
python3 -B PROJECT/scripts/catalog/validate_tool_catalog.py
```

架构测试使用模拟后端，验证状态、上下文、预算与故障边界，不需要下载大模型，也不代表真实模型质量评测。

真实模型推理入口：

```bash
python3 PROJECT/scripts/agent/run_agent.py run --session demo --interactive
python3 PROJECT/scripts/agent/run_agent.py inspect --session demo
```

运行前需按配置准备基座模型、LoRA adapter、Embedding / Reranker 和检索索引。源码发布包不包含这些训练产物；缺少产物时，不能把模拟测试作为真实推理结果。

## 目录结构

```text
PROJECT/
├── agent/        # 上下文编排、状态、记忆与模型接口
├── configs/      # 工具目录与模型 / 检索 / 评测配置
├── data/         # 基础数据与工具语料，不包含大规模中间缓存
├── scripts/      # 数据构建、训练、检索、评测与审计
├── specs/        # 工具、状态、事件和评测数据契约
├── tests/        # 无模型架构测试
├── finetune/     # 训练数据 / 输出目录，运行时生成，不随源码发布
└── reports/      # 已有文本基线报告与边界说明
```

## 结果与可复现范围

- 文本实验细节见 [PROJECT/README.md](PROJECT/README.md)。
- 原始说明保存在 [文本基线说明](PROJECT/reports/00_原始文本工具基线说明.md)。
- 现有质量报告覆盖 100 工具合成闭集，不外推为开放域多模态效果或真实工具执行成功率。
- 已有报告指标与本次源码整理分别看待；新增结果应提供配置、数据来源、预测和计算口径。

## 来源与整理范围

项目由电商导购 Agent 课程 / 实验材料整理，参考 [电商 Agent 项目说明](https://ryannnice.github.io/nexus/ecommerce-agent/)。保留原有实验报告与来源信息，不将参考实现和已有训练结果视为本次整理的新成果。

源码发布范围不包含课程 PPT、录音、付费面试资料、个人简历、API 密钥、本地运行状态、训练权重或大体积推理缓存。
