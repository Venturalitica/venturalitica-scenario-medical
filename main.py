#!/usr/bin/env python3
"""
Medical Spine Segmentation Scenario
Orchestrator for running base (pure ML) vs venturalitica (governance-instrumented) versions.

Usage:
    python main.py --scenario base                    # Run base evaluation (no governance)
    python main.py --scenario venturalitica          # Run governance-instrumented pipeline
    python main.py --scenario compliance             # Run compliance audit only
    python main.py --scenario full-pipeline          # Run complete pipeline with governance
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Medical Spine Segmentation - Base vs Venturalitica Comparison"
    )
    parser.add_argument(
        "--scenario",
        choices=["base", "venturalitica", "compliance", "full-pipeline"],
        default="base",
        help="Which scenario to run",
    )
    parser.add_argument(
        "--model-path",
        help="Path to MONAI model bundle",
    )

    args = parser.parse_args()

    # Add the scenario modules to sys.path
    scenario_dir = Path(__file__).parent
    base_path = scenario_dir / "base_medical"
    vl_path = scenario_dir / "venturalitica_medical"
    shared_path = scenario_dir / "shared_data"

    # Default model path
    model_path = args.model_path or str(
        shared_path / "models" / "wholeBody_ct_segmentation"
    )

    if args.scenario == "base":
        sys.path.insert(0, str(base_path))
        from model_evaluation import main as base_main

        print("=" * 60)
        print("Running Base Evaluation (Pure ML, No Governance)")
        print("=" * 60)
        base_main(model_path=model_path)

    elif args.scenario == "venturalitica":
        sys.path.insert(0, str(vl_path))
        sys.path.insert(0, str(base_path))
        from model_evaluation import main as base_main

        print("=" * 60)
        print("Running Venturalitica Pipeline (With Governance)")
        print("=" * 60)
        base_main(model_path=model_path)

    elif args.scenario == "compliance":
        sys.path.insert(0, str(vl_path))
        from compliance_suite import main as compliance_main

        print("=" * 60)
        print("Running Compliance Audit")
        print("=" * 60)
        compliance_main()

    elif args.scenario == "full-pipeline":
        sys.path.insert(0, str(vl_path))
        sys.path.insert(0, str(base_path))
        print("=" * 60)
        print("Running Full Pipeline: Evaluation + Compliance")
        print("=" * 60)
        from model_evaluation import main as base_main

        base_main(model_path=model_path)
        print("\n" + "=" * 60)
        print("Compliance Audit")
        print("=" * 60)
        from compliance_suite import main as compliance_main

        compliance_main()


if __name__ == "__main__":
    main()
