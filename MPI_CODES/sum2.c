#include <mpi.h>
#include <stdio.h>
void main(int argc, char **argv)
{
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   //
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); // total no of processor
    int n = 12 / nprocs;
    int offset;
    int B[n];
    int global_sum = 0, sum = 0;

    int A[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // printf("printing on processor %d out of %d \n", rank, size);
    // int sum1 = 0;
    // int sum2 = 0;
    // int sum3 = 0;
    // int sum4 = 0;

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
        for (int i = 1; i < nprocs; i++)
        {
            int temp = 0;
            MPI_Recv(&temp, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_sum = global_sum + temp;
        }
        printf("global sum is::%d\n", global_sum);
    }

    else
    {
        MPI_Recv(&B, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < n; i++)
        {
            sum = sum + B[i];
        }
        MPI_Send(&sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    printf("sum from process %d is %d\n", rank, sum);
    MPI_Finalize();
}
