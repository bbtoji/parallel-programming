#include <mpi.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

Matrix read_matrix(const std::string& filename, std::size_t& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file >> n;
    Matrix matrix(n, std::vector<double>(n));

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            file >> matrix[i][j];

    return matrix;
}

void write_matrix(const std::string& filename, const Matrix& matrix) {
    std::ofstream file(filename);
    std::size_t n = matrix.size();

    file << n << "\n";
    for (const auto& row : matrix) {
        for (double v : row)
            file << v << " ";
        file << "\n";
    }
}

// обычное умножение (локальная часть)
void multiply_block(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    std::size_t rows,
    std::size_t n)
{
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            std::cerr << "Usage: mpirun -np P ./app A.txt B.txt result.txt\n";
        MPI_Finalize();
        return 1;
    }

    std::size_t n = 0;
    std::vector<double> A, B, C;

    int rows_per_proc;
    std::vector<int> sendcounts, displs;

    auto start = MPI_Wtime();

    // --- root читает данные ---
    if (rank == 0) {
        std::size_t nA, nB;
        Matrix A2 = read_matrix(argv[1], nA);
        Matrix B2 = read_matrix(argv[2], nB);

        if (nA != nB) {
            throw std::runtime_error("Matrix sizes do not match");
        }

        n = nA;

        A.resize(n * n);
        B.resize(n * n);
        C.resize(n * n, 0.0);

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                A[i * n + j] = A2[i][j];
                B[i * n + j] = B2[i][j];
            }
    }

    // broadcast размера
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // broadcast B (всем нужен)
    if (rank != 0)
        B.resize(n * n);

    MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- распределение строк ---
    rows_per_proc = n / size;
    int extra = n % size;

    int local_rows = (rank < extra) ? rows_per_proc + 1 : rows_per_proc;

    std::vector<int> recvcounts(size), displs_rows(size);

    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = (i < extra) ? rows_per_proc + 1 : rows_per_proc;
            recvcounts[i] = rows * n;
            displs_rows[i] = offset;
            offset += rows * n;
        }
    }

    std::vector<double> A_local(local_rows * n);

    MPI_Scatterv(
        A.data(),
        recvcounts.data(),
        displs_rows.data(),
        MPI_DOUBLE,
        A_local.data(),
        local_rows * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    std::vector<double> C_local(local_rows * n, 0.0);

    // --- вычисление ---
    multiply_block(A_local, B, C_local, local_rows, n);

    // --- сбор результата ---
    MPI_Gatherv(
        C_local.data(),
        local_rows * n,
        MPI_DOUBLE,
        C.data(),
        recvcounts.data(),
        displs_rows.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    auto end = MPI_Wtime();

    if (rank == 0) {
        Matrix C2(n, std::vector<double>(n));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                C2[i][j] = C[i * n + j];

        write_matrix(argv[3], C2);

        std::cout << "Matrix size: " << n << "x" << n << "\n";
        std::cout << "Processes: " << size << "\n";
        std::cout << "Execution time: " << (end - start) << " sec\n";
    }

    MPI_Finalize();
    return 0;
}