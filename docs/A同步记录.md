**成员 A 交付同步记录 — 2026-11-18**

已将 A 的交付同步到当前项目，供 B 后续真实数据检索与联调使用。

- 仓库：<https://github.com/Jie-Leee/graph-agent-gnn-qa>
- 来源分支：`codex/data`
- 来源提交：`640e0d95927c69c7472e6abd1883a66528e10854`
- 比较基线 `origin/main`：`c249ada79321ee8a71a7a9bb08849b8c53c46808`
- 交接入口：[A数据交接](A数据交接.md)、[A收尾与验收](A收尾与验收.md)。

从 A 相对 main 的变更导入了 46 个文件，并合并根 README 中的 A/B 说明。现有 B 检索实现、公共 schemas、查询/评估入口和 smoke demo 均保留。同步前的本地文件另有备份；备份及原始下载文件不属于分支交付内容。

**本地已具备的内容**

| 内容 | 位置与规模 |
|---|---|
| 固定 QA 与问题专用输入 | `data/processed/`：训练 2000、验证 200、测试 300 条，模板 v4、种子 42 |
| 数据清单、统计与样例 | `data/processed/manifest.json`、`stats.json`、`stats.md`、`samples.md` |
| 全组质量标记及语义初审 | `data/quality/`，保留 A 的原始交付 |
| 数据处理模块、下载/检查脚本和测试 | `src/graph_agent/data/`、`scripts/`、`tests/` |
| 原始图谱 | `data/raw/FB15K-237/`：train 272115、valid 17535、test 20466 条边 |
| 实体及关系名称 | `data/raw/entity2text.txt`、`data/raw/relation2text.txt` |

Microsoft 完整 ZIP 下载缓慢，改从 A 已锁定的 KG-BERT 提交 `f3f260309c5e2986467a66e0e68fa7f53b1a150e` 获取图谱文件副本和名称映射。三份图谱分别来自该提交的 `train.tsv`、`dev.tsv`、`test.tsv`，按 A 的布局保存为 `train.txt`、`valid.txt`、`test.txt`。五份文件逐字节的 SHA256 全部与 A 原始数据清单一致，没有转换内容或重新生成 QA。实际获取地址与哈希见 `data/raw/download_mirror.json`；A 的固定来源约定见 `download_sources.json`。

**本次验证结果**

- 46 个直接导入的文件与远程版本逐字节一致；合并前后核对了 B 相关文件哈希。
- 九份固定数据/统计成果和八个数据模块文件通过 manifest 哈希校验。
- `python tests/test_data_pipeline.py`：12/12 通过。
- `python -m unittest discover -s tests -v`：70/70 通过。
- B 的离线 smoke demo 和此前 28 项基础检查通过；已知缺陷仍由独立探针复现，不代表已修复。
- `python scripts/check_data.py`：2500 条 QA 检查通过；未发现跨划分查询组、重复问题 ID、空答案、遗漏完整答案集或不属于可见图的证据等违规。
- `python scripts/accept_data_handoff.py --output outputs/a_handoff_local.json`：当前 B 已跑通 300 题；索引恰为 272115 条训练边，38001 条留出专属边未进入索引，所有返回证据均属于训练图。

本地交接结果单独存于 `outputs/a_handoff_local.json`，没有覆盖 A 原有的 `data/quality/handoff_acceptance.json`。交接脚本使用已给主题 ID 且未加载名称映射，当前实现的平均候选答案节点召回为 51.0716%，全部答案节点均出现的题目比例为 44.3333%；这只是该接口设置的诊断，不是 QA 准确率，也不是 B 的最终质量验收。A 记录的 44.4389% 来自旧检索框架，不能当作当前实现的预期值。

当前目录最初是没有提交历史的本地副本。同步 A 时设置了 `origin` 并获取远程引用，然后按 A 的文件变更同步；当时没有本地提交，因此该次交接报告的 `repository_revision_at_execution` 为 `unavailable`，A 的准确来源提交以本记录为准。后续准备 B 分支提交时，以该 A 提交为基础恢复本地分支关系，将 B 的改动单独提交到 `codex/retrieval`，保留 A 的原始提交历史。

后续建索引只读取 `train.txt`。推理使用 `*.questions.jsonl`，标准答案/关系路径/标准证据只留在训练或评测层。B 的已知问题、召回优化和正式效率报告继续按 [成员 B 功能验收报告](成员B功能验收报告.md) 处理；D 的实际 Agent 仍待实现。
