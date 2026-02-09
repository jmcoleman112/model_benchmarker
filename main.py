
from bench_args import parse_args
from benchmark import run_benchmark


def main() -> int:
    args = parse_args()
    output_path = run_benchmark(args)
    print("Wrote benchmark results to {0}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
