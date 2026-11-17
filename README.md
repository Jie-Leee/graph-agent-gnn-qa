# Graph Agent GNN 知识图谱问答

仓库：https://github.com/Jie-Leee/graph-agent-gnn-qa

分工文档：[五人分工与一周安排 Word](docs/五人分工与一周安排.docx) · [在线阅读分工](docs/五人分工与一周安排.md)

高级机器学习课程五人项目：基于 FB15k-237，比较纯大模型、Agent 加图谱文本检索、Agent 加图谱检索与 GNN 三种问答方案。

**当前状态：A 数据管线与收尾材料已在 `codex/data` 交付。** 已提供可校验下载、两跳模板 QA 构造、分组划分与完整答案/证据审计，2000/200/300 条训练/验证/测试数据成品已提交到 [data/processed](data/processed/)。另提供全查询组质量标记、77组模型语义初审和A侧交接检查；数据边界见 [收尾与验收](docs/A收尾与验收.md)。A 的运行与接口见 [数据交接](docs/A数据交接.md)。

**D Agent 已完成可独立开发的实现。** 包括 LangChain/DeepSeek 客户端、自主工具循环、三种模式接口及离线演示。D 尚未使用真实密钥验证 API，B 正式检索、C 训练模型及 E 正式实验待联调；没有正式三组实验结果。D 的运行方法和边界见 [Agent 交接](docs/D-Agent交接.md)。

## 快速开始

A 数据可独立运行，无需第三方库或 API 密钥：

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/check_data.py
```

需要 Python 3.10 或以上。先验证无需第三方依赖的检索示例：

```bash
python scripts/smoke_demo.py
```

示例使用仓库内的虚构微型图谱，展示“电影 → 导演 → 出生地”的两跳检索及原始证据，不调用 LLM，也不运行 GNN。

实验使用已有的 Anaconda `langchain` 环境：

```bash
conda activate langchain
python --version
# 当前环境已安装 LangChain；缺少依赖时再安装
python -m pip install -r requirements-agent.txt
# C 同学开发 GNN 时再安装
python -m pip install -r requirements-gnn.txt
```

激活环境后，可以先运行不调用模型的脚本响应演示：

```bash
python scripts/agent_demo.py --offline
python scripts/check_agent_routing.py --dry-run
```

复制 `.env.example` 为本地 `.env` 后填写 DeepSeek 密钥（已有文件不要覆盖）。真实密钥只保存在本地，不提交到 GitHub。然后运行 `python scripts/agent_demo.py` 验证真实模型路由，或 `python scripts/run_agent.py --mode llm --limit 1` 开发验证集单题。正式实验入口仍由 E 完成，目前会明确退出提示：

```bash
python scripts/run_experiments.py --config configs/experiment.json
```

## 五人职责与分支

主分支 `main` 保存组长验收后的内容；另外设置五个人的工作分支，总计六个分支。A—E 为占位称呼，E 是组长，可在分工文档中填写姓名。

| 成员 | 工作分支 | 主要负责目录 | 交付内容 |
|---|---|---|---|
| A 数据 | `codex/data` | `src/graph_agent/data/`、`data/` | 图谱清洗、实体名称、QA 构造与划分、数据统计 |
| B 图检索 | `codex/retrieval` | `src/graph_agent/retrieval/` | 实体定位、带方向与关系的局部子图、证据路径、召回与效率 |
| C GNN | `codex/gnn` | `src/graph_agent/gnn/` | GAT、训练、候选打分、五组配置消融 |
| D Agent | `codex/agent` | `src/graph_agent/agent/` | LLM API、自主工具决策、三种模式、演示 |
| E 组长及实验评测 | `codex/evaluation` | `src/graph_agent/evaluation/`、`scripts/run_experiments.py` | 评测代码、三组对比实验、消融结果汇总、图表与失败分析、最后整合 |

E 直接负责评测代码与批量实验入口，统计答案正确率或集合 F1、检索召回、证据支持情况和耗时，组织三组对比，配合 C 汇总消融结果、画图并分析失败案例。C 负责 GNN 各配置的训练，E 在评测和分析之外完成最终联调整合。

公共接口和跨模块依赖先沟通。每人制作自己约 3 页 PPT，汇报约 5 分钟；完成后一起过一遍展示。

## 技术主线

```text
问题 → LLM 决定查询或直接回答 → 图谱局部检索
     → GAT 编码及候选打分 → 真实证据转结构化上下文 → LLM 回答
```

- 默认本地 CPU 训练小型 GAT，大模型调用 API；优先完成两跳问答。
- 主配置：两层 GAT，隐藏维度 128。保留关系类型、边方向和主题实体标记。
- 用训练 QA 监督候选打分；节点表示与图级聚合表示必须真实参与打分。
- 将筛选后的候选与真实图谱路径放入上下文，称为“GNN 引导的证据融合”。若课程要求连续图向量直接注入 LLM，应在开始前确认并调整方案。
- 测试检索和 Agent 输入不包含答案或标准关系路径；禁止按答案补候选节点。

## 实验范围

三组基线：`llm`、`text`、`gnn`。固定相同 LLM、问题集、检索器、子图规模和文本预算。聚合消融指图级 pooling，不能将其说成更换 GAT 的邻居注意力机制。

| 配置 | GAT 层数 | 图级聚合 |
|---|---:|---|
| 主配置 | 2 | mean |
| 层数消融 | 1 | mean |
| 层数消融 | 3 | mean |
| 聚合消融 | 2 | sum |
| 聚合消融 | 2 | max |

每种配置分别训练，验证集选参数，测试集用于最终报告。指标包括答案集合 F1/准确率、候选召回率、缺乏图谱证据的作答比例、工具调用次数、耗时和 token 用量。另准备约 20—30 题路由测试，展示 Agent 的查询与跳过决策，不混入主 QA 指标。

## 完成时间与展示

项目在一周内完成，统一在周五之前交齐。大家按自己的节奏推进，做完就发到自己的 branch 里，附上运行方法和需要对接的内容，群里说一声，组长收到后安排联调。涉及数据、接口和模型输出的依赖提前沟通。

每人约 3 页 PPT、汇报约 5 分钟，共 25 分钟。顺序为 A 任务与数据、B 检索、C GNN、D Agent 与演示、E 实验结果与分析。

## 项目目录

```text
configs/experiment.json       三基线与五组消融配置
data/README.md                数据约定，成品 QA 已提交，原始大包按需下载
docs/                        分工 Word、数据链接与协作说明
scripts/smoke_demo.py         无网络的检索示例
scripts/agent_demo.py         D 的虚构图谱演示，支持离线脚本/真实 API
scripts/run_agent.py          D 的验证集小批量开发入口
scripts/check_agent_routing.py D 的独立合成路由检查
scripts/run_experiments.py    正式实验入口，等待 E 完成
src/graph_agent/schemas.py    公共数据结构
src/graph_agent/data/         A
src/graph_agent/retrieval/    B，含基础有向 BFS 示例
src/graph_agent/gnn/          C
src/graph_agent/agent/        D
src/graph_agent/evaluation/   E
```

协作流程见 [协作指南](docs/协作指南.md)，数据来源见 [数据集链接](docs/数据集链接.md)。
