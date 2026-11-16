# 数据放置与格式

- `raw/`：官方图谱、名称映射、老师提供的 QA，仅保存在本地。
- `processed/`：生成的训练、验证、测试 JSONL，仅保存在本地。
- 仓库提交处理脚本、说明和少量虚构示例，不提交原始大数据。

每条 QA 建议保存如下结构（以下为虚构示例）：

```json
{"question_id":"sample-001","text":"Where was the director of the sample film born?","topic_entity_ids":["film:sample"],"answer_ids":["city:sample"],"relation_path":["directed_by","place_of_birth"],"evidence":[["film:sample","directed_by","person:director"],["person:director","place_of_birth","city:sample"]]}
```

推理只接收 question_id、text 和通过实体定位得到的 topic_entity_ids。answer_ids、relation_path、evidence 是训练监督及评测字段，不传给测试检索或提示词。

无老师 QA 时，在固定可见图上构造约 2000 条训练、200 条验证、300 条测试，以两跳为主。按“主题实体与关系序列”分组划分并保留全部答案，记录种子、可见图边范围和模板版本。该方案评估已知知识库问答，不是原始链接预测协议。
