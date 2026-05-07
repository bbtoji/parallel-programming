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

SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCESSES = [1, 2, 4, 8]

CSV_FILE = Path("mpi_benchmark.csv")
PLOT_FILE = Path("mpi_speedup_plot.png")


@dataclass
class BenchmarkEntry:
    size: int
    procs: int
    time: float | None


def create_matrix(size: int, path: Path) -> None:
    matrix = np.random.uniform(-5, 5, (size, size))

    with path.open("w") as f:
        f.write(f"{size}\n")
        for row in matrix:
            f.write(" ".join(f"{v:.8f}" for v in row) + "\n")


def parse_time(output: str) -> float | None:
    # ищем строку: Execution time: X sec
    pattern = r"([0-9]+\.[0-9]+)"

    for line in output.splitlines():
        if "Execution time" in line or "time" in line.lower():
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))

    return None


def run_mpi(p: int, a: Path, b: Path, r: Path) -> tuple[int, str]:
    if p <= 4:
        process = subprocess.run(
            ["mpirun", "-np", str(p), str(EXECUTABLE), str(a), str(b), str(r)],
            capture_output=True,
            text=True,
        )
    else:
        process = subprocess.run(
            ["mpirun", "--oversubscribe", "-np", str(p), str(EXECUTABLE), str(a), str(b), str(r)],
            capture_output=True,
            text=True,
        )

    return process.returncode, process.stdout


def execute_case(size: int, procs: int) -> BenchmarkEntry:
    file_a = Path(f"A_{size}.txt")
    file_b = Path(f"B_{size}.txt")
    file_r = Path(f"C_{size}_{procs}.txt")

    print(f"\nSize={size}, Procs={procs}")

    create_matrix(size, file_a)
    create_matrix(size, file_b)

    code, output = run_mpi(procs, file_a, file_b, file_r)

    if code != 0:
        print("Ошибка MPI выполнения")
        return BenchmarkEntry(size, procs, None)

    t = parse_time(output)

    if t is None:
        print("Не удалось распарсить время")
        return BenchmarkEntry(size, procs, None)

    print(f"Time: {t:.6f} sec")

    return BenchmarkEntry(size, procs, t)


def save_csv(entries: list[BenchmarkEntry]) -> None:
    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Processes", "Time"])

        for e in entries:
            writer.writerow([e.size, e.procs, f"{e.time:.6f}" if e.time else "N/A"])


def plot_results(entries: list[BenchmarkEntry]) -> None:
    valid = [e for e in entries if e.time]
    if not valid:
        print("Нет данных")
        return

    plt.figure(figsize=(10, 6))

    for p in PROCESSES:
        xs = []
        ys = []

        for size in SIZES:
            for e in valid:
                if e.size == size and e.procs == p:
                    xs.append(size)
                    ys.append(e.time)

        if xs:
            plt.plot(xs, ys, marker="o", label=f"{p} processes")

    plt.xlabel("Размер матрицы (N x N)")
    plt.ylabel("Время выполнения (с)")
    plt.title("MPI: время выполнения vs размер матрицы")
    plt.grid(True)
    plt.legend()

    plt.savefig(PLOT_FILE)


def main() -> None:
    if not EXECUTABLE.exists():
        raise RuntimeError("MPI executable not found")

    results = []

    print("MPI benchmark start")

    for size in SIZES:
        for p in PROCESSES:
            entry = execute_case(size, p)
            results.append(entry)

    save_csv(results)
    plot_results(results)

    print("\nDone")


if __name__ == "__main__":
    main()
