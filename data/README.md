# A 数据模块：运行与格式

负责人：beihaigit233（GitHub 登录账号 `2664718715`），工作分支 `codex/data`。
完整交接说明见 [A数据交接](../docs/A数据交接.md)，真实统计见 [数据说明](../docs/数据说明.md)。

## 从零复现

Python 3.10 及以上，仅用标准库；数据阶段不需要 LLM、PyTorch 或 API 密钥。
在仓库根目录依次运行：

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/check_data.py
```

下载脚本从 Microsoft 获取官方 ZIP（约 146 MB），只提取三份图谱；实体和关系名称
来自固定 KG-BERT 提交。所有原始文件均校验 SHA256。已有正确文件直接离线复用；
原始文件校验不一致会报错，需要明确加 `--force` 才替换。来源记录位于
`data/raw/download_sources.json`。

原始文件布局如下；手动准备时也支持文本映射放在 `FB15K-237/` 内，或五份文件平铺。

```text
data/raw/
  FB15K-237/train.txt       # 272115 条
  FB15K-237/valid.txt       # 17535 条
  FB15K-237/test.txt        # 20466 条
  entity2text.txt           # 14951 条名称映射
  relation2text.txt         # 237 条关系映射
```

## 交付文件

| 文件 | 用途 | Git |
|---|---|---|
| `processed/train.jsonl` | 2000 条训练监督 | 本地生成 |
| `processed/validation.jsonl` | 200 条验证金标 | 本地生成 |
| `processed/test.jsonl` | 300 条测试金标 | 本地生成 |
| `processed/{split}.questions.jsonl` | 相应划分的推理输入，不含答案、标准路径、证据 | 本地生成 |
| `processed/manifest.json` | 全部参数、输入/输出/代码 SHA256 | 本地生成 |
| `processed/stats.json`、`stats.md`、`samples.md` | 详细统计和 12 条可读样例 | 本地生成 |
| `docs/数据说明.md`、`数据样例.md`、`数据清单.json` | 自动复制的交接报告及哈希清单 | 提交 |

原始大文件、完整生成 QA 和缓存按仓库约定不提交，队友运行以上三条命令即可重建。
`--no-docs` 可禁止自动覆盖报告。

## QA 格式与安全读取

金标 JSONL 每行包含：`question_id`、`text`、`topic_entity_ids`、`answer_ids`、
`relation_path`、`evidence`。`evidence` 是 `[head, relation, tail]` 数组列表。
推理 JSONL 仅保留前三个字段。通过现有公共 dataclass 对接，不改 `schemas.py`：

```python
from pathlib import Path
from graph_agent.data.dataset import load_examples, load_questions
from graph_agent.data.loader import load_raw

graph = load_raw(Path("data/raw"))
visible_triples = graph.train_triples
train = load_examples(Path("data/processed/train.jsonl"))  # QAExample
test_inputs = load_questions(Path("data/processed/test.questions.jsonl"))  # Question
test_gold = load_examples(Path("data/processed/test.jsonl"))  # 只交给评测
```

导入前可用 `python -m pip install -e .` 安装本项目，或将 `src/` 加入 `PYTHONPATH`。
检索器与 Agent 只接收 `Question`。不向它们传入 `QAExample`、答案、标准关系路径或
标准证据，也不能按标准答案补候选。提供的主题 ID 属于**已知主题设置**；如果 B
测试实体定位，应另从 `text` 预测 ID，报告定位成功率，不能把给定 ID 当作预测。

## 协议与限制

- 可见图只用原始 `train.txt`；原始 `valid.txt/test.txt` 不用于构造题目或答案。
- 问题是明确标注的**自构两跳模板 QA**，覆盖 24 个白名单关系；其余关系仍可检索。
- 标准答案是沿指定关系序列经过所有中间实体可达的完整集合。含自环或回到主题的
  路径整组排除，不删除其中个别答案；答案超过上限时也整组排除。
- 按 `(主题实体, 关系序列)` 分组划分，组内问法不跨划分，问句全局去重。
  实体、答案和证据边允许共享，因此不宣称实体不相交、边不相交或未知事实泛化。
- 四种题面均要求完整答案集，不依据金标答案数量切换单复数，避免泄漏答案规模。
- 只沿实际存储方向出题，不自动添加反向边。不能将 FB15k-237 删除反向关系冗余
  解释为所有具体关系都无法反向表述。
- 模板与关系白名单限制了语言、关系和任务多样性；这不是天然多跳 QA 基准。

默认运行固定种子 42，生成 2500 条、1250 个独立查询组。每组两种问法属于重复
测量，建议 E 同时报组级结果。全图可达不保证预算有限的检索能召回，详见交接文档。

## 自定义与验证

```bash
python scripts/prepare_data.py --seed 42 --targets train:2000,validation:200,test:300
python scripts/prepare_data.py --paraphrases 4 --targets train:2000,validation:200,test:300
python scripts/prepare_data.py --max-answers 20 --max-groups 60000 --min-path-support 25 --max-dominant-share 0.6 --max-answer-set-share 0.4
python tests/test_data_pipeline.py
python -m unittest discover -s tests -v
python scripts/smoke_demo.py
python scripts/check_data.py --retrieval-diagnostic
```

目标条数必须是每组问法数（1–4）的倍数。构造前校验参数；答案、证据或划分审计
不通过时不写输出。`check_data.py` 另核对文件哈希、代码版本、完整可达答案及
推理文件字段；输入或数据模块变化后应重新构造。代码哈希先将 CRLF 归一为 LF，
避免不同平台 Git 换行设置导致误报。
