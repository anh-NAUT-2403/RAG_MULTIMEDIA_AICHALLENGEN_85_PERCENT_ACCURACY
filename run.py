from __future__ import annotations

import argparse
import json
from pathlib import Path

from keyframe_agent import KeyframeSearchPipeline, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive AIC keyframe retrieval for Google Colab"
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    parser.add_argument("--query", help="Vietnamese visual query")
    parser.add_argument("--query-file", help="UTF-8 text file containing the query")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Build CLIP contact sheets without calling OpenAI",
    )
    parser.add_argument(
        "--task-type",
        choices=("auto", "kis", "trake", "qa"),
        default="auto",
        help=(
            "Task routing: kis for one-frame search, trake for ordered E1-En "
            "events, qa for five video-level evidence bundles"
        ),
    )
    parser.add_argument(
        "--qa-mode",
        choices=("auto", "single_frame", "short_window", "long_range_temporal"),
        default="auto",
        help=(
            "Internal QA route. auto lets the planner decide; long_range_temporal "
            "scans a denser timeline and preserves full-query video candidates"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = args.query
    if args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8").strip()
    if not query:
        raise SystemExit("Provide --query or --query-file")
    config = load_config(args.config)
    pipeline = KeyframeSearchPipeline(config)
    result = pipeline.run(
        query,
        retrieval_only=args.retrieval_only,
        task_type=args.task_type,
        qa_mode=args.qa_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
