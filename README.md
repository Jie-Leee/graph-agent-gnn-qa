# Graph Agent GNN 知识图谱问答

仓库：https://github.com/Jie-Leee/graph-agent-gnn-qa

分工文档：[五人分工与一周安排 Word](docs/五人分工与一周安排.docx) · [在线阅读分工](docs/五人分工与一周安排.md)

高级机器学习课程五人项目：基于 FB15k-237，比较纯大模型、Agent 加图谱文本检索、Agent 加图谱检索与 GNN 三种问答方案。

**当前状态：项目初始框架。** 已提供模块接口、一个不联网的两跳图检索示例、实验配置和协作说明。尚未下载正式数据、接入大模型、训练 GNN 或获得实验结果；示例不能作为正式实验。

## 快速开始

需要 Python 3.10 或以上。先验证无需第三方依赖的检索示例：

```bash
python scripts/smoke_demo.py
```

示例使用仓库内的虚构微型图谱，展示“电影 → 导演 → 出生地”的两跳检索及原始证据，不调用 LLM，也不运行 GNN。

开始正式开发时：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# C 同学开发 GNN 时再安装
python -m pip install -r requirements-gnn.txt
```

复制 `.env.example` 为本地 `.env` 后填写配置。真实密钥只保存在本地，不提交到 GitHub。正式实验入口尚未实现，会明确退出提示：

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
| E 组长 | `codex/evaluation` | `src/graph_agent/evaluation/`、`scripts/run_experiments.py` | 统一评测、实验图表、PR 审核、最终整合 |

公共接口 `src/graph_agent/schemas.py` 和实验配置由组长协调修改，避免各自改成不兼容版本。每人制作自己约 3 页 PPT 并汇报约 5 分钟；组长统一样式与衔接。

## 技术主线

```text
问题 → LLM 决定查询或直接回答 → 图谱局部检索
     → GAT 编码及候选打分 → 真实证据转结构化上下文 → LLM 回答
```

- 默认本地 CPU 训练小型 GAT，大模型调用 API；优先完成两跳问答。
- 主配置：两层 GAT，隐藏维度 128。保留关系类型、边方向和主题实体标记。
- 用训练 QA 监督候选打分；节点表示与图级聚合表示必须真实参与打分。
- 将筛选后的候选与真实图谱路径放入上下文，称为“GNN 引导的证据融合”。若课程要求连续图向量直接注入 LLM，应第一天确认并另行调整方案。
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

## 一周里程碑与展示

| 时间 | 验收目标 |
|---|---|
| 第 1 天 | 确认配套 QA 和融合要求，冻结公共接口，10 题检索到回答闭环 |
| 第 2 天 | 交付数据划分、文本基线、GNN 小样本训练 |
| 第 3 天 | 三种模式首次完整联调，验证集排查漏检与泄漏 |
| 第 4 天 | 完成五组 GNN 配置与验证，固定实验设置 |
| 第 5 天 | 正式测试、结果图与失败分析，冻结功能 |
| 第 6 天 | 拼接约 15 页 PPT，完整排练两次，录制备用演示 |
| 第 7 天 | 缓冲、阻断问题修复、备份与展示验收 |

汇报顺序：A 任务与数据（0—5 分钟），B 检索（5—10），C GNN（10—15），D Agent 与演示（15—20），E 实验与结论（20—25）。

## 项目目录

```text
configs/experiment.json       三基线与五组消融配置
data/README.md                数据放置约定，正式数据不入 Git
docs/                        分工 Word、数据链接与协作说明
scripts/smoke_demo.py         无网络的检索示例
scripts/run_experiments.py    正式实验入口，等待 E 完成
src/graph_agent/schemas.py    公共数据结构
src/graph_agent/data/         A
src/graph_agent/retrieval/    B，含基础有向 BFS 示例
src/graph_agent/gnn/          C
src/graph_agent/agent/        D
src/graph_agent/evaluation/   E
```

协作流程见 [协作指南](docs/协作指南.md)，数据来源见 [数据集链接](docs/数据集链接.md)。
