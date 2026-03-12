from dataclasses import dataclass
from pathlib import Path
import subprocess
import numpy as np
import csv
import re
import matplotlib.pyplot as plt

EXECUTABLE = Path("./matrix_mult")
SIZES = [200, 400, 800, 1200, 1600, 2000]

CSV_FILE = Path("benchmark_results.csv")
PLOT_FILE = Path("benchmark_plot.png")


@dataclass
class BenchmarkEntry:
    size: int
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
        if "Время" in line or "time" in line.lower():
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))

    return None


def run_program(a: Path, b: Path, result: Path) -> tuple[int, str]:
    process = subprocess.run(
        [str(EXECUTABLE.resolve()), str(a), str(b), str(result)],
        capture_output=True,
        text=True
    )

    return process.returncode, process.stdout


def execute_case(size: int) -> BenchmarkEntry:
    file_a = Path(f"A_{size}.txt")
    file_b = Path(f"B_{size}.txt")
    file_res = Path(f"C_{size}.txt")

    print(f"\n--- Размер {size} ---")

    create_matrix(size, file_a)
    create_matrix(size, file_b)

    code, output = run_program(file_a, file_b, file_res)

    if code != 0:
        print("Ошибка выполнения")
        return BenchmarkEntry(size, None)

    exec_time = parse_time(output)

    if exec_time is None:
        print("Не удалось извлечь время")
        return BenchmarkEntry(size, None)

    print(f"Время: {exec_time:.6f} сек")

    return BenchmarkEntry(size, exec_time)


def save_csv(entries: list[BenchmarkEntry]) -> None:
    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Time"])

        for e in entries:
            writer.writerow([
                e.size,
                f"{e.time:.6f}" if e.time else "N/A"
            ])


def plot_results(entries: list[BenchmarkEntry]) -> None:
    valid = [e for e in entries if e.time]

    if not valid:
        print("Нет данных для графика")
        return

    sizes = [e.size for e in valid]
    times = [e.time for e in valid]

    plt.figure(figsize=(9, 5))
    plt.plot(sizes, times, marker="o")
    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (с)")
    plt.title("Зависимость времени умножения от размера")
    plt.grid(True)

    plt.savefig(PLOT_FILE)


def main() -> None:
    if not EXECUTABLE.exists():
        raise RuntimeError("Исполняемый файл matrix_mult не найден")

    results = []

    print("Запуск серии экспериментов")

    for size in SIZES:
        entry = execute_case(size)
        results.append(entry)

    save_csv(results)
    plot_results(results)

    print("\nРезультаты:")

    for r in results:
        t = f"{r.time:.6f}" if r.time else "N/A"
        print(f"{r.size:6d}  {t}")


if __name__ == "__main__":
    main()
