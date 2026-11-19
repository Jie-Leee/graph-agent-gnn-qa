# D/B 检索联调记录

这是B接入阶段的历史记录。C正式权重已于后续接入并通过验证，最新状态与交付材料
见 [D Agent完成说明](D-Agent完成说明.md)。

日期：2026-11-18。当前工作分支为 `codex/agent`，使用已有 Anaconda
`E:\conda\envs\langchain` 环境。B 的 `7f53e06`、`c696f25` 两个提交已合并；
合并提交为 `f196a3b`。README 冲突已保留双方的功能与运行说明。

## 已完成的对接

- `GraphRetrievalTool.from_raw` 以272115条train边建索引，传入完整实体/关系名称。
  valid/test原始边不进入索引，A的固定QA、数据模块和B检索源码保持不变。
- `run_agent.py --mode text` 使用B正式交付的GraphIndex；默认反向遍历开启，
  `--no-allow-inverse` 可关闭。该设置和同一份检索计划应在text/gnn之间固定。
- 使用B的 `retrieve_with_stats`，把检索耗时、展开量、节点/边数、截断标记、
  训练图/名称表哈希和检索源码哈希记入日志。最终模型标识也从API响应元数据记录。
- 保留 `PathStep` 方向信息，提示词中三元组仍为真实的head/relation/tail，
  source_id、target_id、is_reverse单独描述遍历方向；不把反向遍历改写成伪造边。
- 不同跳数的检索结果未必相互包含，最终使用最近一次查询的子图，不合并成超出
  节点/边预算的图；重复查询即使命中缓存，也会重新选定该次查询对应的子图。
- 主QA仍使用A给定的主题ID。可选的无主题输入会保留B定位出的ID；该路径的名称
  歧义仍需单独评测，不能把给定ID实验当作实体定位实验。

## 200道验证题的离线接口与覆盖诊断

复现命令：

```powershell
conda activate langchain
python scripts/check_data.py
python scripts/accept_agent_retrieval.py --gold data/processed/validation.jsonl
```

该命令使用真实图谱、真实检索器和真实Agent循环，但模型替换为明确标注的脚本
提供器：固定请求两跳、随后停止、最终不生成答案。它不测模型自主路由或QA准确率。
所有问题推理结束后，才单独读取gold计算覆盖；标准答案和路径未传入Agent。

同一检索设置：2跳、200节点、800边、allow_inverse=True；证据预算始终为2000个
UTF-8字节。这是原Agent的保守预算，不是精确2000 token。系统提示词及题面不计入
该证据字节预算，API会独立报告实际token用量。

| 覆盖范围（逐题答案节点召回的平均值） | 结果 |
|---|---:|
| B返回子图中的答案节点 | 71.14% |
| D所有排序候选中的答案节点 | 71.14% |
| 初次接入、名称重复展开后的上下文 | 19.24% |
| 压缩名称字典并处理词形后的上下文 | 32.90% |
| 再选择相关真实路径后的最终上下文 | 46.98% |

最终平均每题4.355个上下文候选，全部200题均在2000字节证据预算内。整个过程中
B检索算法、输入图谱和图规模上限未改变；D仅调整推理侧固定文本排序和序列化。
候选覆盖仍明显低于检索覆盖，有限预算和固定文本排序仍是限制，不能宣称完整召回。

压缩格式 `compact_paths_v2` 共享实体/关系名称表，避免每条边重复展开名称。
固定文本排序去除功能词并处理常见词形；在B返回子图内检查真实的一跳/两跳路径，
按关系词匹配选择证据，保留原边方向。不使用金标选择路径，也不从图外补节点。
这一步必要的原因是：B的最短路径可能是语义不相关的捷径，节点被检索到不代表
它的默认显示路径就能回答问题。

本轮仅使用验证集开发这些策略，未用测试集调参。冻结数据检查通过且没有重新
生成QA。测试集原有文件只参与固定数据完整性审计，没有运行测试问答实验。

## 真实 DeepSeek 小规模验证

```powershell
python scripts/agent_demo.py --trace-dir logs/agent-b-integration/live-demo
python scripts/run_agent.py --mode text --limit 1 --output outputs/agent-b-integration/deepseek_text_final.jsonl --trace-dir logs/agent-b-integration
```

微型演示使用虚构图谱，模型真实给出：电影导演出生地问题调用工具1次并返回
Sample City及两条真实证据；2+2不调用工具，回答4。该演示不是FB15k-237主QA成绩。

真实验证题为默认顺序的 `validation-00000`，不是从测试集挑选的展示题。问题要求
Robert Louis Stevenson出生地中包含的全部地点，图内答案为Heriot-Watt University
和University of Edinburgh。这里的“地点”按A固定图谱协议解释。

初次接入与仅压缩上下文时，模型把城市关联的足球队误当成contains关系的答案，
两次F1均为0。失败原因包括预算损失、无关默认路径及模型误读关系，不应只归因于
模型幻觉。增加子图内相关路径选择后，同一题返回完整正确ID集合及真实contains
证据，EM/F1均为1；未把标准答案或标准路径放入提示词。

最终单次运行：模型响应标识deepseek-flash，工具调用1次，输入1854 tokens、输出
244 tokens，含建好索引后的Agent流程耗时约4.00秒。该计时不含下载、图谱加载和
建索引；单次耗时不能代表稳定延迟。该题在验证集上经历开发回归，不能据此推断
全验证集/测试集问答准确率，更不能当作三组对比结果。

本地原始记录（按仓库约定不提交）：

- `outputs/agent-b-integration/offline_acceptance.json`：最终200题诊断、源码/数据哈希。
- `outputs/agent-b-integration/offline_acceptance_before_compaction.json`：初次诊断。
- `outputs/agent-b-integration/offline_acceptance_compact_shortest_paths.json`：压缩后中间诊断。
- `outputs/agent-b-integration/live_smoke_summary.json`：同一真实验证题三次开发记录。
- `logs/agent-b-integration/`：实际提示词、模型响应、检索统计、token与引用证据。

## 后续依赖

最终回归：Anaconda langchain环境中163项unittest全部通过（A 70、B 50、D 43），
原有12项数据管线检查通过；数据全量审计无违规，离线检索和Agent演示通过。
新增回归覆盖train-only名称接入、反向路径、截断统计、非嵌套查询预算、缓存查询
选择、可选实体定位、方向元数据校验、紧凑上下文和无关捷径的证据替换。

C需交付主配置的训练权重、配套实体/关系索引及预处理、完整加载和rank实现。
与C对接时应统一候选证据路径准备策略，再比较GNN排序；否则结果可能同时混入
证据路径变化。gnn模式仍明确要求真实scorer，未使用文本分数或随机分数冒充模型。

E需统一三组实验的模型、检索/反向遍历设置、上下文预算和统计口径。最终应同时
报告检索覆盖、预算后的上下文覆盖、答案指标及错误样本，不能把覆盖率当准确率。
24道合成路由题尚未全量真实运行，其结果也不能混入主QA指标。

此次合并已在本地形成提交；D的适配改动和联调记录保留为后续提交内容，未自动推送。
