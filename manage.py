#!/usr/bin/env python3

import argparse
import subprocess
import sys

def run_tests(args):
    """Run the test suite with pytest."""
    test_path = args.path if args.path else "tests"
    cmd = ["pytest", test_path, "-v"]
    
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    if args.failfast:
        cmd.append("-x")
        
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Task Image Similarity Management Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("path", nargs="?", help="Path to test file or directory")
    test_parser.add_argument("--coverage", "-c", action="store_true", 
                            help="Run tests with coverage report")
    test_parser.add_argument("--failfast", "-x", action="store_true",
                            help="Stop on first failure")
    
    args = parser.parse_args()
    
    if args.command == "test":
        sys.exit(run_tests(args))
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()