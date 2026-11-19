# D Agent：当前可独立完成的交付

最新状态与演示材料见 [D Agent完成说明](D-Agent完成说明.md)。C源码和正式权重
均已合入，主配置真实权重及三模式联调已通过；下文保留前期实现细节和历史验证记录。

本分支实现 LangChain ChatOpenAI 的 DeepSeek/OpenAI 兼容客户端、模型自主路由、有界工具循环、三种
证据模式、结构化回答、实体解析、逐次调用日志和开发入口。公共 schemas、A 数据、
B 检索和 C GNN 源码均未由 D 额外修改。2026-11-18 已合并 B 的检索提交及其兼容的
PathStep/Subgraph 扩展。实验统一使用已有 Anaconda `langchain` 环境，不新建 `.venv`。

已接入 B 检索，在真实 DeepSeek 上验证了查询/跳过微型演示及验证集单题。另完成
200题离线D/B接口与覆盖诊断。C主配置已完成接入，E正式实验仍待完成，没有正式三组
实验结果、全量真实路由准确率或备用录像。详见 [D/B联调记录](D-B检索联调.md)。

2026-11-17 在上述 `langchain` 环境验证：103项 unittest、原有12项数据检查均
通过，离线 Agent 演示、检索示例及24题路由文件校验通过。其中5项集成测试使用
真实 LangChain 消息转换和 OpenAI SDK、模拟 HTTP 响应，验证实际请求格式、
工具循环、错误处理和token统计；并非真实 DeepSeek 联网实验。项目根目录未发现
`.venv`，无需删除；没有创建新虚拟环境，也没有改动现有 conda 环境中的包。

## 先运行

本机环境路径为 `E:\conda\envs\langchain`，已安装 Python 3.13.15、langchain 1.2.12、
langchain-core 1.2.18 和 langchain-openai 1.1.11。版本快照在 `requirements-agent.txt`。
只有缺少依赖时才在该环境中执行 `python -m pip install -r requirements-agent.txt`。
仓库根目录执行，离线演示无需密钥或原始图谱：

```powershell
conda activate langchain
python scripts/agent_demo.py --offline
python scripts/check_agent_routing.py --dry-run
python -m unittest discover -s tests -v
```

若终端未初始化 conda，可以直接使用解释器，不影响环境选择：

```powershell
& 'E:\conda\envs\langchain\python.exe' scripts/agent_demo.py --offline
```

真实 API 默认通过 `LangChainClient` 调用 `ChatOpenAI.bind_tools/invoke`，使用
LangChain 消息转换和使用量元数据。AgentRunner 自己控制循环边界和共享检索，
便于固定实验条件。另保留标准库 `OpenAICompatibleClient` 作为可选适配器；
离线演示不导入 LangChain，不会意外调用 API。

离线演示是虚构图谱和脚本响应，验证查询/跳过流程及真实路径的传递，不证明模型
具有自主决策能力。token 未报告时为 0，日志 `usage_reported=false`，不能称为免费
模型调用或真实 token 实测。真实决策只能以联网运行的消息和工具日志为证。

## DeepSeek 配置和真实调用

复制 `.env.example` 为本地 `.env`（已有文件不要覆盖），只在本机填写密钥：

```dotenv
LLM_API_BASE=https://api.deepseek.com
LLM_API_KEY=填写你自己的密钥
LLM_MODEL=deepseek-flash
LLM_TIMEOUT_SECONDS=60
```

地址和模型默认值按 [DeepSeek 官方文档](https://api-docs.deepseek.com/) 核对。
也支持带 `/v1` 的兼容服务基址。模型名可自行配置；官方 DeepSeek 请求显式关闭
thinking，以便控制路由与回答预算。客户端不使用流式输出；默认 temperature=0、
max_tokens=1024。不同实验必须保持相同模型及参数，记录供应商实际服务的模型版本。

环境变量优先于所选文件。简化 `.env` 读取支持注释行、KEY=VALUE、成对引号；
不展开变量，不支持多行值或行尾注释。也可用环境变量 `DEEPSEEK_API_KEY`。
日志包含题面、模型可见的上下文、模型响应和使用量，不记录请求密钥或 HTTP 响应
错误正文。生成文件写入已忽略的 `logs/`、`outputs/`。

```powershell
# 真实模型 + 虚构微型图谱，同时展示查询和跳过（实际决策由模型给出）
python scripts/agent_demo.py

# 先用 A 的验证集单题开发；默认 limit=1，避免意外批量花费
python scripts/run_agent.py --mode llm --limit 1

# 24 道独立路由题；查询12、跳过12，包含同义问法，不作泛化能力结论
python scripts/check_agent_routing.py --limit 24
```

纯 LLM 可以不下载图谱先调用，但没有实体名称映射时，只有文本/原始名称结果，
不能直接计算 ID 级 QA 指标。要对齐实体 ID，可下载 A 原始文件后使用：

```powershell
python scripts/download_data.py
python scripts/run_agent.py --mode llm --entity-map data/raw/entity2text.txt --limit 1
python scripts/run_agent.py --mode text --limit 1
```

text 命令通过 `GraphRetrievalTool.from_raw` 接入 B 的 GraphIndex，并传入实体/关系
名称；索引只用 train 边。默认允许反向遍历，可用 `--no-allow-inverse` 关闭。设置需
在 text/gnn 间固定。日志包含图谱与检索源码哈希、实际检索耗时和截断标记。
A 的完整验收依赖原始图谱，可另运行
`python scripts/accept_data_handoff.py`；D 源码变更后该脚本会诚实报告需要下游检查，
不会自动调用联网 Agent 或把 A 侧探针当成实际 Agent 验收。

## 流程与公平性

`llm` 只把题目 text 发给模型，不发主题 ID、图谱、候选列表、名称映射或标准答案。
名称映射只在生成完成后解析输出，统一规范化 Unicode、大小写和空白；不做模糊
匹配、不删 -GB/-US 后缀。同名多 ID 的名称保留为歧义，除非模型明确给出合法 ID，
不会将多个同名 ID 合并或全部补入答案。未解析名称、被拒 ID 留在日志中供 E 分析。

`text/gnn` 共用证据模式无关的路由阶段：模型可直接停止或调用 `query_graph`，
参数仅为检索跳数，始终使用给定主题。工具返回实际节点/边数和剩余次数，模型再
决定继续查询或停止。路由默认最多2次工具调用，达到上限后直接进入回答阶段；
单轮批量调用超过剩余额度、未知工具和非法参数均明确失败，不会静默增加预算。

同一个 runner 对同一个完整 Question 缓存路由计划和检索结果。再运行另一证据
模式时复用计划；`text` 用固定关系词汇重合排序，处理常见词形并去除功能词。
在 B 已返回的子图内选择最相关的最多两跳真实路径，避免把无关的最短捷径当成
答案依据；不会增加子图外的节点/边，也不使用标准路径。`gnn` 调 C 的真实 scorer。
回答阶段重新构造消息，只放该模式的候选及证据，不把 text 的回答或路由对话带入
gnn。查询循环因此不依赖 GNN 排序结果；这是控制变量设计，应在报告中说明。
重复的相同跳数查询也复用子图，逻辑工具调用次数仍按模型实际请求计数。最终
上下文使用最近一次查询的子图；不同跳数的检索未必嵌套，不把它们合并成超预算
子图。无给定主题时保留 B 定位出的主题；主QA仍使用给定ID，不宣称实体定位成绩。

每个模式的 Prediction token/耗时包含其共享路由成本，缓存命中时计入原路由成本，
避免把热缓存误报成推理提速。trace 的 `actual_*` 则仅统计本次真实执行的调用和
墙钟耗时。两种口径不可相加作为账单；HTTP 重试可能产生供应商未返回的额外计费。
耗时不包括日志落盘，缺失 usage 会标记未知。每次新实验及模型/数据/参数改变后
新建 runner，或调用 `clear_cache()`；缓存不写磁盘，不跨进程复用。runner 为顺序
执行设计，E 并发实验时每个 worker 单独建实例。

候选上下文默认按 UTF-8 字节数限制到配置的 context_token_budget 数值，作为
保守的、不依赖分词器的预算；不是精确2000 token。固定 JSON 外壳也计入，证据
路径只整条保留，放不下时跳过。当前 `compact_paths_v2` 把实体与关系名称存到共享
字典，路径只引用ID及原始三元组，并保留 source_id/target_id/is_reverse；不重复
展开每条边两端的名称。字节上限只约束证据部分，不约束系统提示词和题面；
实际输入 token 用量由 API 报告。E 若改用精确分词，应对 text/gnn 统一修改预算。

所有候选必须属于检索子图，证据必须在子图内、连通给定主题并到达该候选；允许
沿原边逆向遍历，但序列化和引用始终保留真实 head/relation/tail。最终引用只能
来自实际展示的证据。该检查只验证来源，不代表已自动验证自然语言答案受证据
支持；支持性指标由 E 评测。模型 JSON 格式错误、截断、伪造引用均报错并留日志，
不伪造成功预测。LangChain/OpenAI 客户端配置最多重试2次，实际重试状态按其SDK
规则执行；标准库备用客户端对408/429/指定5xx及网络失败最多重试2次，401等不重试。

## B/C/E 接口

```python
from pathlib import Path
from graph_agent.agent import AgentRunner, AgentSettings, EntityCatalog
from graph_agent.agent.provider import LangChainClient, LLMSettings
from graph_agent.agent.retrieval import GraphRetrievalTool
from graph_agent.data.dataset import load_questions

retriever, catalog = GraphRetrievalTool.from_raw(Path('data/raw'))
runner = AgentRunner(
    LangChainClient(LLMSettings.from_env(Path('.env'))),
    retriever=retriever,  # B适配器，支持allow_inverse并保留检索统计
    scorer=scorer,        # C: rank(Question, Subgraph) -> list[Candidate]，须已训练
    catalog=catalog,
    settings=AgentSettings(), trace_dir=Path('logs/agent'),
)
question = load_questions(Path('data/processed/validation.questions.jsonl'))[0]
text_prediction = runner.run(question, 'text')
gnn_prediction = runner.run(question, 'gnn')  # 同一份路由和检索
```

无 C 模型时 `gnn` 明确拒绝，不使用随机打分或把文本排序冒充 GNN。开发 CLI
现已支持 llm/text/gnn/all，gnn与all通过 `--checkpoint` 指定C训练包中的best.pt。
默认使用 `checkpoints/gat_2_mean/best.pt`；真实权重联调已通过，后续接入E正式实验入口。
B 必须保证索引只有允许的训练边，返回原方向三元组、候选节点和真实路径。
C 的候选排序还会按分数降序、ID破同分，节点与图级表示如何进入分数由 C 自证。
与 C 联调时还需统一证据路径准备策略，避免把“路径选择变化”混入 GNN 排序对比；
可复用 `text_rank` 生成的固定候选路径，再由 C 仅替换候选分数。

E 可以直接读取原有 Prediction；trace 额外提供检索节点、排序后候选、上下文候选、
被拒 ID/歧义名称、缓存情况、实际计费口径和完整提示词。候选召回应区分检索节点、
排序结果与预算后上下文；错误样本不能从总分母静默删除。原始 answer_text 可能
包含未解析名称，应结合 resolution 记录分析，不能仅因 ID 为空宣称模型拒答。

D 只接收精确的 Question，拒收 QAExample。A 的200条验证问题已用离线提供器
通过实际 runner 输入链路；另以替换金标的毒化样本检查实际提示词。未修改冻结
数据或把 data/quality、标准路径、答案、标准证据接入推理。24道合成路由题的
expected_query 只留在检查脚本，绝不传入模型；它们和主 QA 指标分开报告。

当前待交付条件：配合 E 统一正式实验口径后运行完整三组对比。C主配置权重、
加载/候选打分已接入，可用三模式真实日志及已完成的3页PPT和备用录像；单题调试成功
与覆盖诊断均不能替代正式实验结果。离线D/B验收可复现为：

```powershell
python scripts/accept_agent_retrieval.py --gold data/processed/validation.jsonl
```

该命令的路由和回答由脚本提供，不调用模型。仅推理完成后才加载gold计算覆盖率。
