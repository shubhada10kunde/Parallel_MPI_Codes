#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;
#include <cmath>
// Distributed System
int main(int argc, char **argv)
{
    int rank, size;
    int n;
    int A[4][4] = {{1, 5, 0, 4}, {2, 6, 5, 0}, {0, 7, 3, 2}, {4, 0, 1, 1}};
    int x[4] = {1, 2, 4, 6};
    // considering square matrices ...row major form
    n = 4;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sendCounts[n];

    int *recvbuf = new int[sendCountRows];
    // send (rows/size) rows OF A to each process
    MPI_Scatter(&A, sendCountRows, MPI_INT, recvbuf, sendCountRows, MPI_INT, 0, MPI_COMM_WORLD);
    // recvcount == sendcount

    int sCnt = (rows / size);
    int *rcv = new int[sCnt];
    // send sCnt elements of x to each process
    MPI_Scatter(&x, sCnt, MPI_INT, rcv, sCnt, MPI_INT, 0, MPI_COMM_WORLD);
    // gather all the component of X on each processor
    // int gCnt = (size)*sCnt;
    int *xGathered = new int[rows];
    MPI_Allgather(rcv, sCnt, MPI_INT, xGathered, sCnt, MPI_INT, MPI_COMM_WORLD);
    int y[sCnt];
    for (int i = 0; i < size; ++i)
    {
        if (rank == i)
        {
            for (int j = 0; j < sCnt; ++j)
                y[j] = 0;

            for (int k = 0; k < sendCountRows; ++k)
            {
                y[(k / rows)] += recvbuf[k] * xGathered[k % rows];
            }
            // cout << y[0] << endl;
            // cout << y[1] << endl;
        }
    }
    int *bGathered = new int[rows];
    // MPI_Gather collects data from all processes in a given communicator and concatenates them in the given buffer on the specified process.
    // The concatenation order follows that of the ranks.
    MPI_Gather(&y, sCnt, MPI_INT, bGathered, sCnt, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        for (int i = 0; i < rows; ++i)
        {
            cout << "y" << i << " is " << bGathered[i] << endl;
        }
    }
    MPI_Finalize();
    return 0;
}