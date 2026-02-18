#!/usr/bin/env python3
"""SceneSmith entry point for Claude Code mode.

Runs the 5-stage scene generation pipeline using Claude Code subagents
and the SceneSmith MCP server.

Usage:
    python run_scenesmith.py "Create a modern bedroom with a king bed and two nightstands"
    python run_scenesmith.py --stage furniture "Add furniture to the existing room"
"""

import argparse
import logging
import sys

from scenesmith_mcp.orchestrator import (
    PIPELINE_STAGES,
    StageOrchestrator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SceneSmith - NVIDIA + Claude Scene Generation"
    )
    parser.add_argument(
        "scene_description",
        type=str,
        help="Natural language description of the scene to generate",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        choices=PIPELINE_STAGES,
        default=None,
        help="Stage to start from (default: floor_plan)",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        choices=PIPELINE_STAGES,
        default=None,
        help="Stage to stop at (default: manipuland)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum critique rounds per stage (default: 3)",
    )
    parser.add_argument(
        "--early-finish",
        type=float,
        default=7.5,
        help="Score threshold for early stage completion (default: 7.5)",
    )
    parser.add_argument(
        "--mcp-server",
        type=str,
        default="scenesmith",
        help="MCP server name (default: scenesmith)",
    )

    args = parser.parse_args()

    logger.info("SceneSmith - NVIDIA + Claude Scene Generation")
    logger.info(f"Scene: {args.scene_description}")

    orchestrator = StageOrchestrator(
        max_critique_rounds=args.max_rounds,
        early_finish_threshold=args.early_finish,
        mcp_server=args.mcp_server,
    )

    results = orchestrator.run_pipeline(
        scene_description=args.scene_description,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for stage, scores in results.items():
        if scores:
            print(f"  {stage}: {scores.total_score:.1f}/10")
            if scores.critical_issues:
                for issue in scores.critical_issues:
                    print(f"    - {issue}")
        else:
            print(f"  {stage}: completed (no scores)")
    print("=" * 60)


if __name__ == "__main__":
    main()
