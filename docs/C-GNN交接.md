# C GNN 交接

本分支基于 B 的检索提交 `c696f25`，沿用 A 的冻结数据和 B 的两跳检索。GNN 训练使用 2,000 道训练题，配置选择使用 200 道验证题。图谱索引只包含 FB15k-237 的训练边。

## 复现训练

需要 Python 3.10+。GPU 训练时先安装与驱动兼容的 PyTorch CUDA 版本。进入仓库后运行：

```bash
python -m pip install -r requirements-gnn.txt
python -m pip install -e .
python scripts/download_data.py
python -m graph_agent.gnn.train --device auto --output outputs/gnn
```

默认运行五组。仅训练主配置时附加 `--name gat_2_mean`。本次运行使用 RTX 4090、Python 3.12.3、PyTorch 2.3.0+cu121、PyG 2.8.1，种子 42、批大小 24、学习率 0.001、每组最多 20 轮、早停耐心值 5。五组共用子图，按验证 MAP@10 选权重。GPU 算子可能带来小幅运行间差异。

重新训练时，输出目录中每个配置的 `best.pt`、`history.csv` 和 `metrics.json` 对应最佳权重、逐轮记录和配置汇总。`summary.json` 记录实验设置、数据哈希和各配置结果。AutoDL 训练的五组最佳权重位于 [`checkpoints/`](../checkpoints/)。

## 交给 Agent 的接口

```python
from pathlib import Path
from graph_agent.gnn.model import GNNScorer

scorer = GNNScorer(Path("checkpoints/gat_2_mean/best.pt"), device="cpu")
candidates = scorer.rank(question, subgraph)
```

`question` 是 `Question`，`subgraph` 是 B 检索或 D 合并后的 `Subgraph`。候选为子图内具有证据路径的非主题实体，按分数降序排列，同分按实体 ID 排序。每个 `Candidate` 携带子图中的原始三元组路径。模型用关系类型、正反向边和主题标记编码图；问题、候选节点及图级聚合表示共同参与打分。标准答案仅用于监督训练和离线验证。

五组最佳权重均在 CPU 上重新加载，各对 5 道真实验证题检查了排序和证据。D 可将 scorer 传给 `AgentRunner` 运行 `gnn` 模式。

## 验证集结果

| 配置 | 轮数 | 最佳轮次 | MAP@10 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| 两层 mean（主配置） | 19 | 14 | 52.74% | 63.88% |
| 一层 mean | 12 | 7 | 39.80% | 54.18% |
| 三层 mean | 20 | 20 | 61.73% | 69.45% |
| 两层 sum | 9 | 4 | 54.43% | 66.74% |
| 两层 max | 20 | 19 | 56.12% | 64.65% |

MAP@10 按问题取平均，单题分母为标准答案数量与 10 的较小值。Recall@10 按题计算前十候选对完整标准答案集合的召回后平均。检索漏掉的答案保留在指标分母里。五组共享的候选召回率为 71.14%，其中 49 道验证题的子图没有正确答案。正式测试集和 Agent 问答指标由 E 联合评测。

从构建子图到五组跑完耗时 218.4 秒，其中子图准备 134.8 秒。见 [各配置汇总](gnn/summary.json)、[逐轮数据](gnn/history) 和 [五组训练曲线](gnn/training-curves.png)。
