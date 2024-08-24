#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 12

void main(int argc, char **argv)
{
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   //
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // total no of processor
    int n = ARRAY_SIZE / nprocs;
    int offset;
    int B[n];
    int global_sum = 0, sum = 0;

    // int A[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    FILE *arrayInputFile, *outputFile;
    int A[ARRAY_SIZE];

    // opening input file for reading
    arrayInputFile = fopen("array.txt", "r");
    if (arrayInputFile == NULL)
    {
        perror("Error while opening the file");
        return exit(1);
    }

    // reading integers from the file
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        if (fscanf(arrayInputFile, "%d", &A[i]) != 1)
        {
            fprintf(stderr, "error while reading integers from file\n");
            fclose(arrayInputFile);
            return exit(1);
        }
    }
    fclose(arrayInputFile);

    // printf("printing on processor %d out of %d \n", rank, size);
    // int sum1 = 0;
    // int sum2 = 0;
    // int sum3 = 0;
    // int sum4 = 0;

    // PARALLEL IMPLEMENTATION
    if (rank == 0)
    {
        for (int i = 1; i < nprocs; i++)
        {
            // distribution of data to no of processors
            offset = n * i;
            MPI_Send(&A[offset], n, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        for (int i = 0; i < n; i++)
        {
            sum += A[i];
        }

        global_sum = sum;

        // Receive the partial sums from other processes
        for (int i = 1; i < nprocs; i++)
        {
            int temp = 0;
            MPI_Recv(&temp, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_sum = global_sum + temp;
        }

        // writing the global sum to the output file
        outputFile = fopen("output.txt", "w");
        if (outputFile == NULL)
        {
            perror("error while opening the o/p file");
            exit(1);
        }
        fprintf(outputFile, "Global sum::%d\n", global_sum);

        fprintf(outputFile, "Local sums from each process::\n");
        fprintf(outputFile, "Sum from Process %d::%d\n", rank, sum);
        for (int i = 1; i < nprocs; i++)
        {
            int temp = 0;
            MPI_Recv(&temp, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            fprintf(outputFile, "Sum from Process %d:: %d\n", i, temp);
        }

        fclose(outputFile);
        printf("global sum is::%d\n", global_sum);
    }

    else
    {
        MPI_Recv(&B, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < n; i++)
        {
            sum = sum + B[i];
        }
        // Sending partial sum to the root process
        MPI_Send(&sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

        // sending the local sum to the root process for printing
        MPI_Send(&sum, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    printf("sum from process %d is %d\n", rank, sum);
    MPI_Finalize();

    // return 0;
}
