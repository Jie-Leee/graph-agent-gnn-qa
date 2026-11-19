# Graph Agent GNN 知识图谱问答

仓库：https://github.com/Jie-Leee/graph-agent-gnn-qa

分工文档：[五人分工与一周安排 Word](docs/五人分工与一周安排.docx) · [在线阅读分工](docs/五人分工与一周安排.md)

高级机器学习课程五人项目。我们使用 FB15k-237 知识图谱，比较三种回答问题的方法：大模型直接回答、先查图谱再回答、先查图谱并用 GNN 筛选证据再回答。

**当前状态：A 数据管线与收尾材料已在 `codex/data` 交付。** 已提供可校验下载、两跳模板 QA 构造、分组划分与完整答案/证据审计，2000/200/300 条训练/验证/测试数据成品已提交到 [data/processed](data/processed/)。另提供全查询组质量标记、77组模型语义初审和A侧交接检查；数据边界见 [收尾与验收](docs/A收尾与验收.md)。A 的运行与接口见 [数据交接](docs/A数据交接.md)。

**D Agent 已接入 A/B/C，并完成真实主配置与三模式小规模联调。** C正式权重位于 `checkpoints/`，默认使用两层mean主配置；200道验证题的离线契约验收通过，固定前5题的15次真实调用全部正常完成。24道合成路由题、三页PPT和三模式备用日志回放已准备。完整测试集结果由 E 单独汇总。运行与待办见 [D Agent完成说明](docs/D-Agent完成说明.md)，前期证据覆盖与失败分析见 [D/B联调记录](docs/D-B检索联调.md)。

B 的检索支持名称映射、双向遍历、方向标记和检索统计，接口与验证结果见 [成员 B 功能验收报告](docs/成员B功能验收报告.md)。

## 快速开始

需要 Python 3.10 或以上，以下命令都在项目根目录运行。先下载原始图谱并检查数据；这两步只需 Python 自带的库，无需大模型密钥：

```bash
python scripts/download_data.py
python scripts/check_data.py
```

问答数据已经生成，放在 `data/processed/`。需要重新生成时，再按 [data/README.md](data/README.md) 使用 `prepare_data.py` 并指定单独的输出目录。

运行下面的检索示例，可以查看“从问题找到图谱证据”的过程。该示例也可以在下载真实图谱前单独运行：

```bash
python scripts/smoke_demo.py
```

示例使用一个虚构小图，沿“电影 → 导演 → 出生地”走两步，输出相关三元组、路径和子图绘制代码。正常输出包含 5 个节点、4 条边。它用于演示检索过程，还不包含大模型回答或 GNN 打分。

D 开发时使用 Anaconda `langchain` 环境，已有该环境可继续使用。当前 E 盘副本的评测环境位于上一级 `.venv`，可直接运行 `..\.venv\Scripts\python.exe`，详见 [E交接](docs/E评测与整合交接.md)。以下是 Anaconda 环境的原有用法：

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

复制 `.env.example` 为本地 `.env` 后填写 DeepSeek 密钥（已有文件不要覆盖）。真实密钥只保存在本地，不提交到 GitHub。然后运行 `python scripts/agent_demo.py` 验证真实模型路由，或 `python scripts/run_agent.py --mode llm --limit 1` 开发验证集单题。E 的批量实验入口已实现，默认仅运行5道验证题；本地密钥留空时不会调用API：

```bash
python scripts/run_experiments.py --config configs/experiment.json
```

## 五人职责与分支

`main` 保存组长审核后的整合版本。五位成员分别在自己的分支开发，完成后交给组长合并。A—E 代表五位成员，E 同时担任组长。

| 成员 | 工作分支 | 主要负责目录 | 交付内容 |
|---|---|---|---|
| A 数据 | `codex/data` | `src/graph_agent/data/`、`data/` | 整理图谱和名称，生成并划分问答数据，提供统计 |
| B 图检索 | `codex/retrieval` | `src/graph_agent/retrieval/` | 找到问题中的实体，检索附近的图，输出证据并评估检索效果 |
| C GNN | `codex/gnn` | `src/graph_agent/gnn/` | 训练 GAT，为候选答案打分，比较五组模型配置 |
| D Agent | `codex/agent` | `src/graph_agent/agent/` | 接入大模型，让 Agent 决定是否查图谱，完成三种回答模式和演示 |
| E 组长及实验评测 | `codex/evaluation` | `src/graph_agent/evaluation/`、`scripts/run_experiments.py` | 编写评测，汇总实验结果和图表，分析错误，整合各模块 |

C 负责训练不同配置的模型；E 负责统一评测、比较结果并整合项目。检索召回率表示标准答案有多少进入候选节点，问答准确率表示最终回答是否正确，两者需要分别统计。

修改公共数据结构或其他成员需要调用的接口前，先与相关成员沟通。每人准备约 3 页 PPT，讲清自己完成的工作，汇报约 5 分钟。

## 技术主线

完整系统计划按下面的流程运行。LLM 指大模型，GAT 是本项目使用的图神经网络：

```text
问题 → Agent 决定直接回答，还是先查图谱
     → B 检索相关节点和证据路径
     → C 用 GAT 给候选答案打分（仅 gnn 模式）
     → 把选出的证据交给大模型 → 生成回答
```

- GAT 默认在本地 CPU 上训练，大模型通过 API 调用。优先处理沿两条关系才能找到答案的问题，即“两跳问答”。
- 主模型使用两层 GAT，隐藏维度为 128。输入保留关系类型、边的方向，以及问题对应的主题实体标记。
- 用训练集中的标准答案学习打分。打分时同时使用候选节点的信息和整张子图汇总的信息。
- GNN 帮助筛选候选答案，再把对应的真实图谱路径整理成文字交给大模型。当前方案不直接把图向量输入大模型；若课程要求这种做法，需要另行确认。
- 检索器和 Agent 只接收问题及主题实体 ID。标准答案和预先标注的路径只用于训练或评测，不能用于给测试题补答案。

## 实验范围

比较三种方案：`llm` 由大模型直接回答；`text` 先查图谱，把证据整理成文字再回答；`gnn` 在检索后增加 GNN 打分，再用筛选出的证据回答。

三种方案使用相同的大模型和问题集。两种使用检索的方案，还要统一检索器、子图大小上限和输入文本长度。

另外比较下面五组 GAT 配置，观察层数和信息汇总方式对结果的影响，这类比较称为“消融实验”。图级聚合是把所有节点的信息汇总成整张子图的信息：`mean` 取平均，`sum` 求和，`max` 取最大值；它与 GAT 内部的邻居注意力计算是两个步骤。

| 配置 | GAT 层数 | 图级聚合 |
|---|---:|---|
| 主配置 | 2 | mean |
| 层数消融 | 1 | mean |
| 层数消融 | 3 | mean |
| 聚合消融 | 2 | sum |
| 聚合消融 | 2 | max |

每种配置都单独训练，用验证集选择参数，最后在测试集上统一比较。记录最终答案的准确率或集合 F1、检索召回率、没有证据支持的回答比例、工具调用次数、耗时和 token 用量。

另准备约 20—30 道题，单独检查 Agent 能否决定“查图谱”或“直接回答”。这部分结果单独展示，不与主要问答指标混在一起。

## 完成时间与展示

按分工约定，项目在一周内完成，周五之前交齐。各成员完成后推送到自己的分支，附上运行方法和需要其他成员配合的事项，再通知组长安排联调。涉及数据、接口或模型输出的问题，提前沟通。

每人约 3 页 PPT、汇报约 5 分钟，总计约 25 分钟。顺序为：A 介绍任务与数据，B 介绍图谱检索，C 介绍 GNN，D 演示 Agent，E 汇报实验结果与分析。

## 项目目录

```text
configs/experiment.json       三种问答方案和五组 GAT 配置
data/README.md                数据下载、格式和使用说明
data/processed/              已生成的问答数据、统计和校验清单
docs/                        分工、交接和验收说明
scripts/download_data.py      下载并校验原始图谱与名称映射
scripts/check_data.py         检查数据是否完整、是否符合约定
scripts/smoke_demo.py         无需下载数据的检索示例
scripts/query_graph.py        查询本地图谱，输出三元组、路径和子图绘制代码
scripts/evaluate_retrieval.py 统计检索召回率、耗时和子图大小
scripts/agent_demo.py         D 的离线/真实 API 微型图谱演示
scripts/run_agent.py          D 的验证集开发入口
scripts/check_agent_routing.py D 的独立路由检查
scripts/accept_agent_retrieval.py D/B 在真实验证问题上的离线接口与覆盖诊断
scripts/run_experiments.py    正式评测、并发和断点恢复入口
src/graph_agent/schemas.py    各模块共用的数据结构
src/graph_agent/data/         A：数据处理
src/graph_agent/retrieval/    B：实体定位、图谱检索与证据输出
src/graph_agent/gnn/          C：模型训练与候选打分
src/graph_agent/agent/        D：大模型调用与 Agent 流程
src/graph_agent/evaluation/   E：评测指标
tests/                       数据与检索的自动检查
```

协作流程见 [协作指南](docs/协作指南.md)，数据来源见 [数据集链接](docs/数据集链接.md)。

## E 统一评测

E 已整合 ABCD 的交付，并完成 300 道测试题、七种设置的真实 DeepSeek 评测。见 [正式报告和图表](docs/evaluation/formal/E实验评测报告.md) 与 [E 五分钟汇报提纲](docs/evaluation/formal/E汇报提纲.md)。请参阅 [E评测与整合交接](docs/E评测与整合交接.md)。默认5题验证联调用于检查接口；完整300题测试须显式选择。模拟预测只验证程序，不是模型成绩。本机源码和环境均保存在 E 盘高级机器学习作业目录，API密钥仅写本地 `.env`。
