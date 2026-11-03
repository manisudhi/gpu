#!/usr/bin/env python3
"""
scripts/run_pipeline.py
Command-line interface to execute the AMR GPU pipeline (all phases or selected phases).
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.amr.pipeline import AMRPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Run GPU-Accelerated AMR Machine Learning Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--phase", type=int, default=0, help="Phase number to run (0 for all phases)")
    args = parser.parse_args()

    config_path = root_dir / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration for {config['project']['name']}")
    pipeline = AMRPipeline(config=config, project_root=root_dir)

    if args.phase == 0:
        summary = pipeline.run_all()
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE - REAL MEASURED RESULTS")
        print("=" * 60)
        for phase, data in summary.items():
            if isinstance(data, dict):
                print(f"\n[{phase.upper()}]:")
                for k, v in data.items():
                    print(f"  {k}: {v}")
            else:
                print(f"\n{phase}: {data}")
    else:
        logger.info(f"Executing selected Phase {args.phase} only...")
        # Individual phase dispatch if desired
        if args.phase == 1:
            pipeline.run_phase_1()
        elif args.phase == 2:
            pipeline.run_phase_2()
        else:
            logger.info("For full reproducibility across dependencies, running run_all()")
            pipeline.run_all()


if __name__ == "__main__":
    main()
