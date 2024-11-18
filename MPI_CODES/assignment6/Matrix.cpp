#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>

const int N = 8; // Size of the matrix
const int P = 4; // Number of processors

void matrixMultiply(const std::vector<int> &A_block, const std::vector<int> &B_block, std::vector<int> &C_block, int blockSize)
{
    for (int i = 0; i < blockSize; ++i)
    {
        for (int j = 0; j < blockSize; ++j)
        {
            int sum = 0;
            for (int k = 0; k < blockSize; ++k)
            {
                sum += A_block[i * blockSize + k] * B_block[k * blockSize + j];
            }
            C_block[i * blockSize + j] = sum;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != P)
    {
        if (rank == 0)
            std::cerr << "This program requires exactly 4 processes.\n";
        MPI_Finalize();
        return -1;
    }

    int blockSize = N / 2; // each will handle a (N/2)x(N/2) block
    std::vector<int> A(N * N);
    std::vector<int> B(N * N);
    std::vector<int> C(N * N, 0);

    if (rank == 0)
    {
        // Read matrices A and B from "input.txt"
        std::ifstream inputFile("input.txt");
        if (!inputFile)
        {
            std::cerr << "Error opening input file\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N * N; ++i)
        {
            inputFile >> A[i];
        }
        for (int i = 0; i < N * N; ++i)
        {
            inputFile >> B[i];
        }

        inputFile.close();
    }

    // Scatter blocks of A and B to each processor
    std::vector<int> A_block(blockSize * blockSize);
    std::vector<int> B_block(blockSize * blockSize);
    std::vector<int> C_block(blockSize * blockSize, 0);

    MPI_Scatter(A.data(), blockSize * blockSize, MPI_INT, A_block.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B.data(), blockSize * blockSize, MPI_INT, B_block.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform block matrix multiplication on each processor
    matrixMultiply(A_block, B_block, C_block, blockSize);

    // Print C_block for each processor to verify submatrices C00, C01, C10, C11
    std::cout << "Processor " << rank << " - Submatrix C" << (rank / 2) << (rank % 2) << ":\n";
    for (int i = 0; i < blockSize; ++i)
    {
        for (int j = 0; j < blockSize; ++j)
        {
            std::cout << C_block[i * blockSize + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Gather the computed C_block from each processor
    MPI_Gather(C_block.data(), blockSize * blockSize, MPI_INT, C.data(), blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Output the result matrix C to "output.txt"
        std::ofstream outputFile("output.txt");
        if (!outputFile)
        {
            std::cerr << "Error creating output file\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        outputFile << "Resultant Matrix C:\n";
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                outputFile << C[i * N + j] << " ";
            }
            outputFile << "\n";
        }
        outputFile.close();
    }

    MPI_Finalize();
    return 0;
}