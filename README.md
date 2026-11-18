# Graph Agent GNN 知识图谱问答

仓库：https://github.com/Jie-Leee/graph-agent-gnn-qa

分工文档：[五人分工与一周安排 Word](docs/五人分工与一周安排.docx) · [在线阅读分工](docs/五人分工与一周安排.md)

高级机器学习课程五人项目：基于 FB15k-237，比较纯大模型、Agent 加图谱文本检索、Agent 加图谱检索与 GNN 三种问答方案。

**当前状态：A 数据交付已同步，B 图检索基础实现已具备。** 已从 `codex/data` 同步可复现的数据处理脚本、2000/200/300 条训练/验证/测试 QA、统计及质量记录。A 的运行与接口见[数据交接](docs/A数据交接.md)，质量边界见[收尾与验收](docs/A收尾与验收.md)。本地保留 B 的实体定位、双向检索、证据路径与评估工具；已知问题见[成员 B 功能验收报告](docs/成员B功能验收报告.md)。尚未接入大模型、训练 GNN 或获得正式三组对比结果。

## 快速开始

需要 Python 3.10 或以上。先验证无需第三方依赖的检索示例：

```bash
python scripts/smoke_demo.py
```

示例使用仓库内的虚构微型图谱，展示实体名称定位、“电影 → 导演 → 出生地”的两跳检索、原始证据、耗时和 Mermaid 子图，不调用 LLM，也不运行 GNN。

运行 A/B 自动检查：`python tests/test_data_pipeline.py`、`python -m unittest discover -s tests -v`。单独运行 `python tests/test_retrieval.py` 还会输出 B 的已知问题探针；当前问题及验收范围见[成员 B 功能验收报告](docs/成员B功能验收报告.md)。

A 的固定版本 QA 已在 `data/processed/`，无需重新生成。首次使用真实图谱时，下载并校验原始文件，再检查交付；这些步骤仅使用 Python 标准库：

```bash
python scripts/download_data.py
python scripts/check_data.py
python scripts/accept_data_handoff.py --output outputs/a_handoff_local.json
```

推理优先读取 `*.questions.jsonl`，标准答案仅用于训练或事后评测；建索引只使用 `data/raw/FB15K-237/train.txt`。若需重新构造 QA，请按 [data/README.md](data/README.md) 指定单独输出目录，保留冻结版本。

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
data/README.md                A 数据与接口约定，固定 QA 已提交，原始图谱按需下载
docs/                        分工 Word、数据链接与协作说明
scripts/smoke_demo.py         无网络的检索示例
scripts/query_graph.py        对本地图谱执行一次查询并输出 Mermaid
scripts/evaluate_retrieval.py 在 QA JSONL 上统计召回和耗时
scripts/run_experiments.py    正式实验入口，等待 E 完成
src/graph_agent/schemas.py    公共数据结构
src/graph_agent/data/         A
src/graph_agent/retrieval/    B，实体链接、双向局部检索、可视化与召回评估
src/graph_agent/gnn/          C
src/graph_agent/agent/        D
src/graph_agent/evaluation/   E
```

### 图谱检索接口（成员 B）

`GraphIndex.from_tsv(...)` 可读取 `head<TAB>relation<TAB>tail` 图谱文件，以及可选的 `ID<TAB>名称` 实体/关系映射。传入 `Question` 已定位的 `topic_entity_ids` 时优先使用这些 ID；为空时会在问题文本中按名称别名进行大小写不敏感的连续词组匹配。此实体链接器是可解释的精确别名匹配，不做语义猜测；歧义别名会返回对应的多个 ID。

```python
from graph_agent.retrieval import GraphIndex, to_mermaid
from graph_agent.schemas import Question

index = GraphIndex.from_tsv(
    "data/raw/FB15K-237/train.txt",
    entity_names_path="data/raw/entity2text.txt",
    relation_names_path="data/raw/relation2text.txt",
)
result = index.retrieve_with_stats(
    Question("q1", "Where was the film director born?", ("/m/film",)),
    max_hops=2, max_nodes=200, max_edges=800,
)
print(result.stats)                       # latency, sizes, and truncation flags
print(to_mermaid(result.subgraph, index.entity_names, index.relation_names))
```

检索支持正向和反向邻接遍历；返回的三元组始终保留图谱原始方向，`Subgraph.path_steps` 的 `is_reverse` 标记路径中反向走过的边。返回结果受跳数、节点数和边数限制，统计会说明是否触发节点或边上限。

用于离线评估时，`benchmark_retrieval(index, cases, ...)` 在每个查询完成后才读取 `RetrievalCase` 中的标准答案与证据，输出答案召回率、证据召回率、平均/P50/P95 延迟、子图规模和截断比例。它不把标准答案或标准路径传给检索器；真实数据的实验统计仍需使用 A/E 提供的数据与评测流程生成。

有本地数据后，也可直接从命令行查询或评估：

```bash
python scripts/query_graph.py --triples data/raw/FB15K-237/train.txt --entity-names data/raw/entity2text.txt --relation-names data/raw/relation2text.txt --question "Where was the film director born?"
python scripts/evaluate_retrieval.py --triples data/raw/FB15K-237/train.txt --qa data/processed/test.jsonl --entity-names data/raw/entity2text.txt --relation-names data/raw/relation2text.txt
```

查询时可重复传入 `--topic-entity ID` 覆盖自动名称匹配；`--directed-only` 关闭反向遍历。评估脚本读取项目约定的 JSONL 中 `question_id`、`text`、`topic_entity_ids`、`answer_ids` 和可选 `evidence` 字段，结果输出到终端，也可用 `--output report.json` 写入文件。

协作流程见 [协作指南](docs/协作指南.md)，数据来源见 [数据集链接](docs/数据集链接.md)。
