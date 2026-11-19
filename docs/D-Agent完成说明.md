# D Agent 交付与验证结果

更新：2026-11-19。当前分支为 `codex/agent`，A/B/C已接入。C源码提交为
`7634ab6`，五组正式权重的发布提交为 `8e73b92`。已使用C原始主配置完成CPU加载、
200道验证题离线契约验收和固定前5题的真实三模式联调，没有使用此前停止的本地复现产物。

## 已完成的D工作

| 交付 | 当前状态 |
|---|---|
| DeepSeek / LangChain | 真实API接入、用量/异常日志、密钥隔离 |
| 自主工具循环 | query_graph、finish或直接结束；最多2次图谱查询 |
| llm/text/gnn | 固定前5道验证题共15次真实调用全部正常完成 |
| GNN主配置 | C的2层、128维、mean模型在CPU加载和推理通过 |
| 公平比较 | text/gnn共用路由、子图、候选与真实路径，仅替换排序分数 |
| 独立路由检查 | 最终24/24流程成功，12查询、12跳过、错误0 |
| 演示材料 | 3页可编辑PPT、约5分钟讲稿、独立HTML回放和备用录像 |
| E对接 | Prediction、权重哈希、实际模型名、候选覆盖、错误行和运行示例 |

最终169项unittest与原有12项数据管线检查通过。合成checkpoint单测只检查结构，
不作效果证据；本轮另外使用C真实权重执行了实际验证。实验使用已有Anaconda
`langchain` 环境（Python 3.13.15、torch 2.15.0+cpu、PyG 2.8.1），不创建 `.venv`。

## 真实权重与完整性

主配置：`checkpoints/gat_2_mean/best.pt`，大小9512078字节。
SHA256：`0444815ce0b921993e84fcf2b2c8c7e2d01891386084d8298e68184453752345`。
本地Git blob与C发布分支均为 `18c135f3b9c75e56be7bb98274a096dd2e2a883d`。
加载后核对：2层、隐藏维度128、mean pooling；14505实体、237关系、1498问题词。
使用torch的weights_only安全加载和CPU映射，无需重新训练或导出CPU版本。

没有因C的三层模型验证分数更高而改选主配置。其余四组正式权重也已随C提交同步，
供E的消融实验使用；D本轮只验证既定主配置。此前暂停的本地中间训练文件仍在
被忽略的 `outputs/gnn-reproduced/`，带有NOT_FOR_EXPERIMENTS标记，未使用或打包。

## 运行方式

```powershell
conda activate langchain
python scripts/agent_demo.py --offline
python scripts/agent_demo.py
python scripts/run_agent.py --mode all --limit 1
python scripts/run_agent.py --mode all --limit 5 --continue-on-error
python scripts/check_agent_routing.py --limit 24
```

默认权重即上述仓库主配置，默认只跑验证集前1题；可用 `--checkpoint PATH`
切换权重。真实API使用本地 `.env`，密钥不会进入预测或日志。原始图需用A脚本
下载并校验。C正式权重是用户授权同步的已发布文件，D不额外提交本地训练产物。
错误样本保留status/error_type，不静默丢弃；有错误时命令仍以非零状态退出。

## 200题离线契约与覆盖诊断

```powershell
python scripts/accept_agent_gnn.py --gold data/processed/validation.jsonl
```

使用真实C主配置和B检索器，由脚本提供路由与最终空回答，不调用LLM。200题全部
确认text/gnn复用相同路由、候选集合及路径策略；真实GNN对同一候选集合打分。
全部推理完成后才加载gold计算覆盖，没有将答案或标准路径送入推理。

| 指标 | text | gnn |
|---|---:|---:|
| 上下文答案节点宏召回 | 46.98% | 54.99% |

两组均为2跳、200节点、800边、allow_inverse=True，证据预算2000个UTF-8字节。
差值约8.01个百分点。它不是QA准确率，也不是精确2000 token实验。B子图答案节点
覆盖率仍为71.14%；两组最终上下文不同只来自候选排序和相同预算下的筛选。

## 固定前5题的真实三模式联调

真实调用固定使用 `validation-00000` 至 `validation-00004`，15次调用全部正常，
text/gnn共享路由日志逐题一致。5题对应3个独立查询组，包含组内改写，而且部分
样例此前用于开发；因此不能作为独立测试、泛化估计或正式方案对比。

这一小批次集合EM/F1为llm 0、text 0.8、gnn 1.0。纯LLM未给出可评测实体ID，
结果也受固定图谱闭集及名称映射协议影响，不能解释为模型完全没有相关知识。
没有从分母中删除失败记录，没有运行主测试集问答，也未执行E的正式批量实验。

差异样例 `validation-00004`：text未将作曲家路径放入最终上下文而没有作答；
gnn将电影music关系到Elliot Goldenthal、再到其出生地Brooklyn的真实路径放入
上下文，模型回答正确。回放展示它是为了观察模式差异，其他4题及全部调用仍保留。

## 独立路由检查

最终显式finish协议为24/24流程成功。首版23/24成功：一道换算题尝试调用未声明的
finish而明确失败。随后正式声明停止工具、验证参数并增加回归测试，再完整运行
24题，未将未知动作直接当作成功。finish是控制动作，不计为图谱查询。
题集是小规模合成流程检查，包含同义问法，不能混入主QA指标或宣称通用路由能力。

## PPT、讲稿与备用回放

- [3页可编辑PPT](agent/D-Agent汇报.pptx)，含讲解备注和原生图表。
- [约5分钟讲稿](agent/D-Agent讲稿.md)。
- `outputs/agent-final/deliverables/Agent三模式真实日志回放.html`：双击打开，
  可翻页或自动播放，不需要网络、密钥或现场加载权重。
- `outputs/agent-final/deliverables/Agent三模式备用演示.webm`：1280×720，约42秒，
  由保存的真实日志回放录制。画面标记非实时API调用，不伪造现场执行。

回放包含差异样例的llm/text/gnn实际输出与算术跳过例子。录制器只打开本项目
独立本地页面，不采集用户桌面或个人浏览器。可复现为：

```powershell
python scripts/build_agent_replay.py --trace-dir logs/agent-final/three-modes --trace-dir logs/agent-final/routing-v2 --include validation-00004 --include route-skip-01 --output outputs/agent-final/deliverables/Agent三模式真实日志回放.html
```

## 交给E的内容与统计口径

E可调用 `AgentRunner.run(Question, mode)`，或参考开发CLI的all模式初始化。
正式入口 `scripts/run_experiments.py` 仍由E完成。统一LLM、检索/反向遍历设置、
证据预算、名称解析和错误分母后，再运行完整正式实验与消融分析。

Prediction包含共享路由的逻辑成本；trace.actual_*只统计本次实际执行的成本，
两者不能相加当账单。缺失usage明确标记未知，trace记录实际模型名及权重哈希。
引用来源检查不等于自然语言支持性证明，最终证据支持指标仍需E评测。

本地记录：`outputs/agent-final/gnn_acceptance.json`、`three_modes.jsonl`、
`three_mode_smoke_summary.json`、`routing.json` 与 `logs/agent-final/`。
新版交付包包含C正式主配置及哈希、源码、必要日志和演示材料，不含 `.env`、
原始大图、任何API密钥或未完成的本地训练权重。其他消融权重从仓库checkpoints获取。
