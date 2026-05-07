#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import subprocess
import numpy as np
import csv
import re
import matplotlib.pyplot as plt


EXECUTABLE = Path("./matrix_mult")

MATRIX_SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS = [1, 2, 4, 8]

CSV_FILE = Path("openmp_benchmark.csv")
PLOT_FILE = Path("openmp_plot.png")


@dataclass
class BenchmarkEntry:
    size: int
    threads: int
    time: float | None


def create_matrix(size: int, path: Path) -> None:
    matrix = np.random.uniform(-5, 5, (size, size))

    with path.open("w") as f:
        f.write(f"{size}\n")

        for row in matrix:
            f.write(" ".join(f"{v:.8f}" for v in row) + "\n")


def parse_time(output: str) -> float | None:
    pattern = r"([0-9]+\.[0-9]+)"

    for line in output.splitlines():
        if "Execution time" in line or "Время" in line:
            match = re.search(pattern, line)

            if match:
                return float(match.group(1))

    return None


def run_program(
    a: Path,
    b: Path,
    result: Path,
    threads: int
) -> tuple[int, str]:

    process = subprocess.run(
        [
            str(EXECUTABLE.resolve()),
            str(a),
            str(b),
            str(result),
            str(threads)
        ],
        capture_output=True,
        text=True
    )

    return process.returncode, process.stdout


def execute_case(size: int, threads: int) -> BenchmarkEntry:
    file_a = Path(f"A_{size}.txt")
    file_b = Path(f"B_{size}.txt")
    file_res = Path(f"C_{size}_{threads}.txt")

    print(f"\nРазмер={size} Потоки={threads}")

    if not file_a.exists():
        create_matrix(size, file_a)

    if not file_b.exists():
        create_matrix(size, file_b)

    code, output = run_program(
        file_a,
        file_b,
        file_res,
        threads
    )

    if code != 0:
        print("Ошибка выполнения")
        print(output)

        return BenchmarkEntry(size, threads, None)

    exec_time = parse_time(output)

    if exec_time is None:
        print("Не удалось получить время")

        return BenchmarkEntry(size, threads, None)

    print(f"Время: {exec_time:.6f} сек")

    return BenchmarkEntry(size, threads, exec_time)


def save_csv(entries: list[BenchmarkEntry]) -> None:
    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "MatrixSize",
            "Threads",
            "ExecutionTime"
        ])

        for e in entries:
            writer.writerow([
                e.size,
                e.threads,
                f"{e.time:.6f}" if e.time else "N/A"
            ])


def plot_results(entries: list[BenchmarkEntry]) -> None:
    plt.figure(figsize=(10, 6))

    for size in MATRIX_SIZES:
        subset = [
            e for e in entries
            if e.size == size and e.time is not None
        ]

        subset.sort(key=lambda x: x.threads)

        threads = [e.threads for e in subset]
        times = [e.time for e in subset]

        plt.plot(
            threads,
            times,
            marker="o",
            label=f"{size}x{size}"
        )

    plt.xlabel("Количество потоков")
    plt.ylabel("Время выполнения (сек)")
    plt.title("OpenMP: зависимость времени от количества потоков")

    plt.xticks(THREADS)

    plt.grid(True)
    plt.legend()

    plt.savefig(PLOT_FILE)

    print(f"\nГрафик сохранён: {PLOT_FILE}")


def main() -> None:
    if not EXECUTABLE.exists():
        raise RuntimeError(
            "Исполняемый файл matrix_mul не найден"
        )

    results: list[BenchmarkEntry] = []

    print("=== OpenMP Benchmark ===")

    for size in MATRIX_SIZES:
        for threads in THREADS:
            entry = execute_case(size, threads)
            results.append(entry)

    save_csv(results)
    plot_results(results)

    print("\nИтоговые результаты:\n")

    for r in results:
        time_str = (
            f"{r.time:.6f}"
            if r.time is not None
            else "N/A"
        )

        print(
            f"Размер={r.size:4d} "
            f"Потоки={r.threads:2d} "
            f"Время={time_str}"
        )


if __name__ == "__main__":
    main()