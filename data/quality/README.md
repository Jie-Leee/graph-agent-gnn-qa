# A 收尾质量记录

本目录针对模板 v4、种子42的冻结 QA，补充可复现质量标记、模型语义初审和
A侧接口检查。原始 `data/processed/` 十个文件未改动。本目录含金标及由金标
推导的标记，只供 A 质量分析、C 训练监督或 E 事后评测；不能送入测试检索和
Agent 提示词。主推理仍读取 `data/processed/*.questions.jsonl`。

| 文件 | 内容 |
|---|---|
| `group_annotations.jsonl` | 1250 个查询组的风险、答案规模、主题出度、同名候选及直邻答案标记 |
| `summary.json` | 全量统计、39 种路径分布、确定性抽样规则、输入与脚本哈希 |
| `review_samples.jsonl` | 77 个不同查询组的题面、名称、答案及证据，覆盖全部39路径 |
| `semantic_review.jsonl` | 与上述77组逐一对应的模型语义初审结论、具体理由和建议 |
| `handoff_acceptance.json` | A导出、当前B框架、D问题输入契约及未完成接入状态 |
| `closeout_manifest.json` | 本次质量产物哈希、审阅范围及真实验收边界 |

`review_samples.jsonl` 中 `pending_semantic_review` 是可重建抽样模板的初始状态；
完成情况以独立 `semantic_review.jsonl` 为准，按 `group_id` 联接，不能只看模板状态。

## 生成与核验

从仓库根目录运行：

```bash
python scripts/download_data.py
python scripts/check_data.py
python scripts/audit_data_quality.py
python scripts/accept_data_handoff.py
python tests/test_data_pipeline.py
python -m unittest discover -s tests -v
```

质量脚本只重建前三个自动产物，不生成或覆盖语义审阅判断。语义记录是针对
固定抽样逐条读取后的模型初审，不是人工专家标签，不保证现实事实真值。
抽样使用种子20261117，按每种关系路径最多两组，优先跨划分；其中38种路径各
两组，1种在已发布数据中只有一组，共77组。审阅记录由两路模型分工覆盖，每组
只审一次，不是双人复审，也没有计算标注者一致性。

`clear` 表示本次未发现明显图内表达问题；`usable_with_caveat` 表示需按已披露的
关系/名称边界解释；`needs_clarification` 表示存在需明确说明的语义或消歧风险。
这是分层选择后的主观描述，不能将状态比例称为全数据错误率或人审通过率。

重新运行交接脚本会记录当时的Git版本；未来D实现变动后，脚本不会自动调用
新Agent或LLM，而会标记需D另行验收。报告里的队友接收状态只有收到实际确认
才能更新。执行脚本成功不等于B/D成员已签收或三种模式的正式实验已通过。

详细规则见 [A数据质量处理约定](../../docs/A数据质量处理约定.md)，收尾结果与
后续职责见 [A收尾与验收](../../docs/A收尾与验收.md)。
