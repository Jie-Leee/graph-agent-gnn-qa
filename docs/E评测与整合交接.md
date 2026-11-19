# E 评测与整合交接

E 负责三基线、五组 GNN 消融的统一批量评测、运行记录、结果图表、错误分析及模块整合。

## 已整合的交付

以 D 分支 `241a40af8d784581377f19170b872b885dcf63f5` 为源码基底。经 GitHub 提交比较，它已包含 A `640e0d9`、B `c696f25` 和 C `8e73b92` 的交付，保留五组真实 GNN 权重。本地项目位于 `E:\高级机器学习作业\graph-agent-gnn-qa`，Python 环境位于上一级 `.venv`；未假定队友电脑上的 Anaconda 环境存在。

## 运行方法

在项目根目录运行，命令里的 `python` 应替换成已装依赖的 Python；本机可直接用 `..\.venv\Scripts\python.exe`。

```powershell
# 安装依赖，CPU即可，不需要重训
python -m pip install -r requirements-evaluation.txt

# 软件流程验证，不调用API，不是实验成绩
python scripts/run_experiments.py --backend mock --limit 5 --variants llm text gat_2_mean gat_1_mean gat_3_mean gat_2_sum gat_2_max --output outputs/evaluation/mock
python scripts/build_evaluation_report.py --predictions outputs/evaluation/mock/predictions.jsonl --output outputs/evaluation/mock/report --allow-mock

# 填好本地.env后，先做5道验证题真实联调
python scripts/run_experiments.py --backend live --split validation --limit 5 --variants llm text gat_2_mean gat_1_mean gat_3_mean gat_2_sum gat_2_max --output outputs/evaluation/live-smoke

# 全量测试：300题，三基线与五种GNN配置共7个变体
python scripts/run_experiments.py --backend live --split test --limit 300 --workers 8 --variants llm text gat_2_mean gat_1_mean gat_3_mean gat_2_sum gat_2_max --output outputs/evaluation/test
python scripts/build_evaluation_report.py --predictions outputs/evaluation/test/predictions.jsonl --output outputs/evaluation/test/report
```

同一命令中断后可追加 `--resume`；数据、题数、变体顺序、配置、模型、权重、源码或Git版本变化时，请使用新的输出目录。恢复以一个问题的全部变体为单位，中断重试仍可能有额外 API 开销。并发发生在题与题之间，同一题的变体仍串行并共享路由。批量评测在保存全部记录后遇到任何失败预测会返回非零，这时先检查清单，再按预定规则保留失败记录并生成报告。

## DeepSeek 配置

根目录 `.env` 只保存在本地，已在 `.gitignore` 中排除，不要提交密钥。已准备平台地址 `https://api.deepseek.com`、请求模型 `deepseek-flash`、超时120秒，由各成员在自己的机器上填密钥。若D使用其他模型，请在正式实验开始前统一。记录请求模型和服务端实际返回模型；模型版本不同不能简单归因于GNN。

## 指标和公平性

- 只把 `Question` 传给 Agent，答案、标准路径、证据金标仅用于训练监督或事后评分。
- 多答案报告完整集合EM及集合P/R/F1；失败题保留并计0，空答与名称映射未完成分开分析。
- 三种模式统一测试题、LLM、检索预算。文本和GNN共用同题路由、候选和证据路径，只改变排序；同题路由缓存必须保持一致。
- 主配置预先固定 `gat_2_mean`；比较1/2/3层及mean/sum/max图级聚合，不根据测试分数选主配置。
- D名为 `context_token_budget` 的参数实际控制 UTF-8 字节；报告写证据预算2000字节，不写精确2000 token。
- 同时报告检索候选与上下文答案召回。API的实际tokens/耗时与包含共享路由的逻辑成本分别统计，不能相加；缺usage保留未知。
- 图谱答案不支持比例仅表示输出实体不在该固定图谱问题的完整金标集合中，是QA错误代理，不直接等同现实事实幻觉。无引用不自动算幻觉，空答另报回答率。
- 当前测试300题来自150个查询组，每组两个改写。报告组级统计及按组的配对bootstrap，不把改写当成独立推理样本。

## 输出

每次运行输出 `manifest.json`、`predictions.jsonl`、`summary.json/csv` 与 `traces/`。报告脚本生成三基线及消融表、中文报告、错误明细和PNG/SVG图。mock、验证子集和全量正式测试有明确标记；未完成真实API实验时，不提供模拟出来的模型结论。

## 解释边界

实体ID集合的EM/F1只评价规范化后的ID，部分名称未能解析时不能据此断言整段自然语言回答完全正确；报告保留未解析、歧义和被拒ID计数及原始trace供人工复核。

本机原始图谱大包下载停滞后，改从KG-BERT作者仓库的锁定提交取得五份原始文件；每份SHA256均与A交付清单完全一致。实际来源与字节数保存在本地 data/raw/E_download_provenance.json，没有更改训练或问答数据。
