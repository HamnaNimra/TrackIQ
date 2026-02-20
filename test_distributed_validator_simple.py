#!/usr/bin/env python3
"""Simple test script for DistributedValidator to check determinism fixes."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trackiq_core.distributed_validator import DistributedValidator, DistributedValidationConfig


def main():
    """Run a simple test of the distributed validator."""
    print("Testing DistributedValidator with recent determinism fixes...")
    print("=" * 60)

    try:
        # Create validator
        validator = DistributedValidator()

        # Use default config
        config = DistributedValidationConfig()
        print(f"Configuration:")
        print(f"  Steps: {config.num_steps}")
        print(f"  Processes: {config.num_processes}")
        print(f"  Tolerance: {config.loss_tolerance}")
        print(f"  Learning rate: {config.learning_rate}")
        print()

        # Run validation
        print("Running validation...")
        results = validator.run_validation(config)

        # Print summary
        summary = results["summary"]
        print("
Results:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Passed steps: {summary['passed_steps']}")
        print(f"  Failed steps: {summary['failed_steps']}")
        print(".2%")
        print(f"  Overall: {'PASS' if summary['overall_pass'] else 'FAIL'}")

        # Check if losses match
        single_losses = results["single_process_losses"]
        multi_losses = results["multi_process_losses"]

        print("
Loss comparison (first 10 steps):")
        print("Step | Single Loss | Multi Loss  | Delta | Match")
        print("-" * 55)

        max_check = min(10, len(single_losses), len(multi_losses))
        matches = 0

        for i in range(max_check):
            single = single_losses[i]
            multi = multi_losses[i]
            delta = abs(single - multi)
            rel_delta = delta / max(abs(single), abs(multi)) if max(abs(single), abs(multi)) > 0 else 0
            match = rel_delta <= config.loss_tolerance
            if match:
                matches += 1

            print("4d")

        print(".1f")

        # Overall assessment
        if summary["overall_pass"]:
            print("\n✅ SUCCESS: Distributed training validation PASSED!")
            print("   Losses match between single and multi-process training.")
        else:
            print("\n❌ FAILURE: Distributed training validation FAILED!")
            print("   Losses do not match within tolerance.")

            # Show some failing examples
            print("\nFailing steps (first 5):")
            for comp in results["comparisons"][:5]:
                if not comp["passed"]:
                    print(".6f")

        return summary["overall_pass"]

    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)