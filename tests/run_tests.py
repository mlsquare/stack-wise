#!/usr/bin/env python3
"""
Test runner for Stack-Wise test suite.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --unit             # Run unit tests only
    python tests/run_tests.py --integration     # Run integration tests only
    python tests/run_tests.py --examples        # Run example tests only
    python tests/run_tests.py --verbose         # Verbose output
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_tests(test_type=None, verbose=False):
    """Run tests based on type."""
    test_dir = Path(__file__).parent
    
    if test_type == "unit":
        test_path = test_dir / "unit"
    elif test_type == "integration":
        test_path = test_dir / "integration"
    elif test_type == "examples":
        test_path = test_dir / "examples"
    else:
        test_path = test_dir
    
    # Find all test files
    test_files = []
    for pattern in ["test_*.py", "*_test.py"]:
        test_files.extend(test_path.glob(pattern))
        if test_type is None:  # If running all tests, also check subdirectories
            for subdir in ["unit", "integration", "examples"]:
                subdir_path = test_path / subdir
                if subdir_path.exists():
                    test_files.extend(subdir_path.glob(pattern))
    
    if not test_files:
        print(f"No test files found in {test_path}")
        return False
    
    # Run tests
    success = True
    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Running {test_file.name}")
        print(f"{'='*60}")
        
        try:
            # Run the test file
            result = subprocess.run([
                sys.executable, str(test_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file.name} passed")
                if verbose and result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
            else:
                print(f"❌ {test_file.name} failed")
                print("STDERR:")
                print(result.stderr)
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
                success = False
                
        except Exception as e:
            print(f"❌ {test_file.name} failed with exception: {e}")
            success = False
    
    return success

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run Stack-Wise tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--examples", action="store_true", help="Run example tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Determine test type
    test_type = None
    if args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.examples:
        test_type = "examples"
    
    print("🧪 Stack-Wise Test Suite")
    print("=" * 60)
    
    if test_type:
        print(f"Running {test_type} tests...")
    else:
        print("Running all tests...")
    
    success = run_tests(test_type, args.verbose)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()


