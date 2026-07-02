import argparse
import os
import sys

# Add project root to path so modules can be found if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="HPatches Local Features Benchmark")
    parser.add_argument('--dataset', type=str, default=r"d:\AdditionalProject\hpatches-sequences-release", help='Path to HPatches dataset')
    parser.add_argument('--output', type=str, default=r"d:\AdditionalProject\project\results", help='Output directory for results and plots')
    
    args = parser.parse_args()
    
    print(f"Starting benchmark...")
    print(f"Dataset path: {args.dataset}")
    print(f"Output path: {args.output}")
    
    runner = ExperimentRunner(args.dataset, args.output)
    
    print("\nMethod Status:")
    for method in runner.methods:
        status = "Available" if getattr(method, 'available', True) else "Unavailable (OpenCV build limitation)"
        print(f"- {method.name}: {status}")
    print("")
    
    runner.run()
    
    print(f"\nBenchmark complete! Results saved in {args.output}")

if __name__ == "__main__":
    main()
