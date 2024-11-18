#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric> // For std::accumulate

#define ARRAY_SIZE 12

int main(int argc, char **argv)
{
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // Total number of processes

    int n = ARRAY_SIZE / nprocs; // Number of elements per process
    int offset;
    std::vector<int> B(n);
    int global_sum = 0, local_sum = 0;

    std::vector<int> A(ARRAY_SIZE);

    if (rank == 0)
    {
        // Root process reads the array from file
        std::ifstream arrayInputFile("array.txt");
        if (!arrayInputFile.is_open())
        {
            std::cerr << "Error while opening the input file!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < ARRAY_SIZE; i++)
        {
            if (!(arrayInputFile >> A[i]))
            {
                std::cerr << "Error reading array values from file!" << std::endl;
                arrayInputFile.close();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        arrayInputFile.close();

        // Distribute portions of the array to other processes
        for (int i = 1; i < nprocs; i++)
        {
            offset = n * i;
            MPI_Send(&A[offset], n, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Calculate local sum for rank 0
        local_sum = std::accumulate(A.begin(), A.begin() + n, 0);
        global_sum = local_sum;

        // Receive partial sums from other processes
        for (int i = 1; i < nprocs; i++)
        {
            int partial_sum = 0;
            MPI_Recv(&partial_sum, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_sum += partial_sum;
        }

        // Write results to output file
        std::ofstream outputFile("output.txt");
        if (!outputFile.is_open())
        {
            std::cerr << "Error while opening the output file!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        outputFile << "Global sum: " << global_sum << "\n";
        outputFile << "Local sums from each process:\n";
        outputFile << "Sum from Process " << rank << ": " << local_sum << "\n";

        // Receive and log local sums from other processes
        for (int i = 1; i < nprocs; i++)
        {
            int local_sum_from_process = 0;
            MPI_Recv(&local_sum_from_process, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            outputFile << "Sum from Process " << i << ": " << local_sum_from_process << "\n";
        }

        outputFile.close();
        std::cout << "Global sum is: " << global_sum << std::endl;
    }
    else
    {
        // Non-root processes receive their portion of the array
        MPI_Recv(B.data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Calculate local sum
        local_sum = std::accumulate(B.begin(), B.end(), 0);

        // Send partial sum to root process
        MPI_Send(&local_sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        // Send local sum for logging
        MPI_Send(&local_sum, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    // Print local sum for each process
    std::cout << "Sum from process " << rank << " is " << local_sum << std::endl;

    MPI_Finalize();
    return 0;
}
