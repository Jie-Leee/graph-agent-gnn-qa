"""Build the FB15k-237 two-hop QA dataset.

Usage
-----
    python scripts/prepare_data.py --raw data/raw --output data/processed

The fixed QA release lives in ``data/processed`` and is versioned alongside
the scripts. Human-readable reports are also copied to ``docs/`` for review.
Use a different --output directory for exploratory variants.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_agent.data.prepare import DEFAULT_TARGETS, prepare_dataset  # noqa: E402
from graph_agent.data.loader import RawDataError  # noqa: E402
from graph_agent.data.split import SplitError  # noqa: E402


def _parse_targets(value: str) -> dict[str, int]:
    """Parse ``train:2000,validation:200,test:300``."""
    targets: dict[str, int] = {}
    for chunk in value.split(","):
        name, _, size = chunk.partition(":")
        name = name.strip()
        if name not in DEFAULT_TARGETS:
            raise argparse.ArgumentTypeError(f"未知划分：{name}")
        if name in targets:
            raise argparse.ArgumentTypeError(f"重复划分：{name}")
        try:
            targets[name] = int(size)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{name} 条数必须是整数") from error
    missing = set(DEFAULT_TARGETS) - set(targets)
    if missing:
        raise argparse.ArgumentTypeError(f"缺少划分：{', '.join(sorted(missing))}")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 FB15k-237 两跳问答数据")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--docs", type=Path, default=ROOT / "docs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=dict(DEFAULT_TARGETS),
        help="形如 train:2000,validation:200,test:300",
    )
    parser.add_argument("--paraphrases", type=int, default=2, help="每组问法数")
    parser.add_argument("--max-answers", type=int, default=20, help="单题答案数上限")
    parser.add_argument("--max-groups", type=int, default=60000)
    parser.add_argument("--min-path-support", type=int, default=25)
    parser.add_argument("--max-dominant-share", type=float, default=0.6)
    parser.add_argument("--max-answer-set-share", type=float, default=0.4)
    parser.add_argument("--no-docs", action="store_true", help="不把报告复制到 docs/")
    args = parser.parse_args()

    try:
        stats = prepare_dataset(
            args.raw,
            args.output,
            args.seed,
            targets=args.targets,
            paraphrases=args.paraphrases,
            max_answers=args.max_answers,
            max_groups=args.max_groups,
            min_path_support=args.min_path_support,
            max_dominant_share=args.max_dominant_share,
            max_answer_set_share=args.max_answer_set_share,
        )
    except (RawDataError, SplitError, ValueError) as error:
        parser.exit(2, f"数据构造失败：{error}\n")

    if not args.no_docs:
        args.docs.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output / "stats.md", args.docs / "数据说明.md")
        shutil.copyfile(args.output / "samples.md", args.docs / "数据样例.md")
        shutil.copyfile(args.output / "manifest.json", args.docs / "数据清单.json")

    audit = stats["audit"]
    print("=== 构造完成 ===")
    for name, info in stats["splits"].items():
        print(
            f"{name:<11} {info['items']:>5} 条 / {info['groups']:>4} 组"
            f" / 平均答案 {info['answers_mean']} / 路径 {info['path_types']} 种"
        )
    print(
        f"候选组 {stats['candidate_groups']} -> 可用组 {stats['kept_groups']}"
        f" -> 实际用 {sum(i['groups'] for i in stats['splits'].values())}"
    )
    print(f"路径类型 {stats['path_count']} 种，被排除路径 {len(stats['dropped_paths'])} 条")
    if audit["violations"]:
        print("!!! 泄漏审计未通过：")
        for item in audit["violations"]:
            print("   -", item)
        raise SystemExit(1)
    print("审计通过：组不跨划分、问句不重复、答案完整、证据可在训练图中连通")
    print(f"输出目录：{args.output}")


if __name__ == "__main__":
    main()
