"""Markdown reporting for the prepared dataset.

The report is the artefact A hands to the rest of the group and to the course
report: it has to state what the data is, how it was built, what was excluded
and why, so that nobody has to reverse-engineer the JSONL.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .surface import RELATION_SURFACE, entity_surface, relation_surface

SPLIT_LABEL = {"train": "训练集", "validation": "验证集", "test": "测试集"}


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_stats_markdown(stats: dict) -> str:
    raw = stats["raw"]
    splits = stats["splits"]
    audit = stats["audit"]

    parts: list[str] = ["# FB15k-237 两跳问答数据说明", ""]
    parts.append(
        "> 本文件由 `scripts/prepare_data.py` 自动生成，请勿手工修改。"
        f" 模板版本 `{stats['template_version']}`，随机种子 `{stats['seed']}`。"
    )
    parts.append("")

    parts.append("## 一、原始图谱")
    parts.append("")
    parts.append(
        _table(
            ["项目", "数值"],
            [
                ["训练三元组", raw["triples"]["train"]],
                ["验证三元组", raw["triples"]["valid"]],
                ["测试三元组", raw["triples"]["test"]],
                ["实体数（全库）", raw["entities_all_splits"]],
                ["关系数（全库）", raw["relations_all_splits"]],
                ["训练集实体数", raw["entities_train"]],
                ["训练集关系数", raw["relations_train"]],
                ["平均出度", raw["out_degree_mean"]],
                ["最大出度", raw["out_degree_max"]],
                ["最大入度", raw["in_degree_max"]],
                ["自环三元组", raw["self_loops_train"]],
            ],
        )
    )
    parts.append("")
    parts.append("文本映射覆盖率：")
    parts.append("")
    parts.append(
        _table(
            ["映射文件", "条目数", "训练集覆盖率"],
            [
                [
                    "entity2text.txt",
                    raw["entity_text_entries"],
                    f"{raw['entity_text_coverage_train']:.2%}",
                ],
                [
                    "relation2text.txt",
                    raw["relation_text_entries"],
                    f"{raw['relation_text_coverage_train']:.2%}",
                ],
            ],
        )
    )
    parts.append("")

    parts.append("## 二、可见图政策")
    parts.append("")
    parts.append(
        "问答只在**训练三元组**构成的可见图上构造。验证集与测试集的三元组不参与"
        "出题，也不作为答案来源。这样每道题的金标答案都能沿可见图正向走到，"
        "评测的是「在已知图谱上问答」，而不是原始链接预测协议。"
    )
    parts.append("")
    parts.append(
        "本数据集只沿原始三元组方向构造两跳题，不自动添加反向边。某类问题能否"
        "构造取决于实际关系及方向；FB15k-237 去除了反向关系冗余，不代表所有"
        "自然语言关系都具有相同的存储方向。"
    )
    parts.append("")

    parts.append("## 三、问答数据规模")
    parts.append("")
    rows = []
    for name in ("train", "validation", "test"):
        info = splits[name]
        rows.append(
            [
                SPLIT_LABEL[name],
                info["items"],
                info["groups"],
                info["paraphrases_per_group"],
                info["answers_mean"],
                info["answers_max"],
                info["path_types"],
            ]
        )
    parts.append(
        _table(
            ["划分", "问答条数", "组数", "每组问法", "平均答案数", "最大答案数", "路径类型数"],
            rows,
        )
    )
    parts.append("")

    parts.append("## 四、答案规模分布")
    parts.append("")
    hist = stats["answer_histogram"]
    parts.append(
        _table(
            ["答案数", "问答条数"],
            [[size, count] for size, count in sorted(hist.items())],
        )
    )
    parts.append("")

    parts.append("## 五、两跳路径分布")
    parts.append("")
    parts.append(
        f"白名单共 {len(RELATION_SURFACE)} 个语义明确的关系，两两组合后实际用到 "
        f"{stats['path_count']} 条路径。下列为问答条数最多的 15 条。"
    )
    parts.append("")
    path_rows = []
    for path, count in stats["path_usage"][:15]:
        path_rows.append(
            [
                relation_surface(path[0]),
                relation_surface(path[1]),
                count,
            ]
        )
    parts.append(_table(["第一跳", "第二跳", "问答条数"], path_rows))
    parts.append("")

    parts.append("## 六、问法模板")
    parts.append("")
    parts.append(
        f"每个组内选 {stats['paraphrases_per_group']} 种不同问法，模板由组键的稳定哈希"
        "决定，组内全部问法必然落在同一划分。所有题面都要求返回完整答案集，"
        "措辞不依据标准答案个数切换。"
    )
    parts.append("")
    parts.append(
        _table(
            ["模板", "使用次数"],
            [[f"`{tpl}`", count] for tpl, count in stats["template_usage"]],
        )
    )
    parts.append("")

    parts.append("## 七、泄漏审计")
    parts.append("")
    parts.append(
        _table(
            ["检查项", "结果"],
            [
                ["同一组跨划分出现", audit["cross_split_groups"]],
                ["同一问句跨划分出现", audit["cross_split_questions"]],
                ["划分内重复问句", sum(audit["duplicated_questions_within_split"].values())],
                ["答案包含主题实体", audit["self_answer_items"]],
                ["空答案", audit["empty_answer_items"]],
                ["证据不能按两跳路径连通答案", audit["disconnected_evidence_items"]],
                ["证据边不在训练图", audit["evidence_not_in_visible_graph_items"]],
                ["标准答案不等于训练图完整可达集合", audit["incomplete_answer_set_items"]],
                ["重复问题 ID", audit["duplicate_question_ids"]],
                ["训练/测试主题实体重叠", audit["topic_entity_overlap_train_test"]],
                ["测试集主题实体未在训练集出现", audit["test_topic_entities_unseen_in_train"]],
                ["训练/测试答案实体重叠", audit["answer_entity_overlap_train_test"]],
            ],
        )
    )
    parts.append("")
    if audit["violations"]:
        parts.append("**发现违规项：**")
        parts.append("")
        for item in audit["violations"]:
            parts.append(f"- {item}")
    else:
        parts.append("全部检查通过：组与问句未跨划分，答案与训练图完整两跳可达集合一致，证据可连通且均在训练图内。")
    parts.append("")
    parts.append(
        "本划分是共享训练图上的查询组划分；实体、答案及部分证据边可以跨 QA 划分共享，"
        "不能据此宣称实体不相交、边不相交或未知事实泛化。提供的主题 ID 属于已知主题"
        "设置；若评估实体定位能力，需另从题面预测主题，不能把给定 ID 当作定位结果。"
    )
    parts.append("")

    parts.append("## 八、被排除的关系路径")
    parts.append("")
    parts.append(
        "「主答案占比」= 最常见单个答案占该路径全部答案的比例；"
        "「重复答案集占比」= 出现次数最多的完整答案集占该路径组数的比例。"
        "两者过高都意味着不查图也能猜到答案。"
    )
    parts.append("")
    dropped = stats["dropped_paths"]
    if dropped:
        parts.append(
            _table(
                ["路径", "组数", "主答案占比", "重复答案集占比", "排除原因"],
                [
                    [
                        f"{relation_surface(p)} >> {relation_surface(q)}",
                        groups,
                        f"{share:.0%}",
                        f"{repeated:.0%}",
                        reason,
                    ]
                    for p, q, groups, share, repeated, reason in dropped[:15]
                ],
            )
        )
    else:
        parts.append("无。")
    parts.append("")

    parts.append("## 九、复现方式")
    parts.append("")
    parts.append("```bash")
    parts.append("python scripts/prepare_data.py --raw data/raw --output data/processed")
    parts.append("```")
    parts.append("")
    parts.append(
        f"随机种子 `{stats['seed']}`、模板版本 `{stats['template_version']}`、"
        f"每组问法数 `{stats['paraphrases_per_group']}`、答案数上限 "
        f"`{stats['max_answers']}`，候选上限 `{stats['max_groups']}`，路径支持下限 "
        f"`{stats['min_path_support']}`，主答案占比上限 `{stats['max_dominant_share']}`，"
        f"重复答案集占比上限 `{stats['max_answer_set_share']}`。原始文件、生成文件和"
        "数据模块代码的 sha256 均记录在 `manifest.json`。"
    )
    parts.append("")
    return "\n".join(parts)


def render_samples_markdown(
    records: dict[str, list[dict]],
    entity_text: dict[str, str],
    limit: int = 4,
) -> str:
    """A handful of fully readable examples, one block each."""

    def label(entity_id: str) -> str:
        name = entity_surface(entity_id, entity_text)
        return f"{name} (`{entity_id}`)" if name != entity_id else f"`{entity_id}`"

    parts = ["# 数据样例（人工可读）", ""]
    parts.append(
        "每块给出一条问答的完整字段，包括证据路径。注意 `answer_ids` 与 `evidence` "
        "只在训练与评测时使用，推理阶段不进入提示词。"
    )
    parts.append("")
    for name in ("train", "validation", "test"):
        items = records.get(name, [])[:limit]
        if not items:
            continue
        parts.append(f"## {SPLIT_LABEL[name]}")
        parts.append("")
        for item in items:
            parts.append(f"**Q（{item['question_id']}）** {item['text']}")
            parts.append("")
            parts.append(f"- 主题实体：{label(item['topic_entity_ids'][0])}")
            parts.append(f"- 关系路径：`{' >> '.join(item['relation_path'])}`")
            answers = "、".join(label(a) for a in item["answer_ids"])
            parts.append(f"- 标准答案（{len(item['answer_ids'])}）：{answers}")
            evidence = "；".join(
                f"{h} -[{r}]-> {t}" for h, r, t in item["evidence"]
            )
            parts.append(f"- 证据：{evidence}")
            parts.append("")
    return "\n".join(parts)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
