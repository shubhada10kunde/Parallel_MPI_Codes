#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

void printMatrix(double A[][5], int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n + 1; ++j)
        {
            cout << A[i][j] << "    ";
        }
        cout << endl;
    }
    cout << endl;
}

void gauss(double localB[], double temp[], int pivot, int m)
{
    double factor = localB[pivot] / temp[pivot];
    for (int i = 0; i < m; ++i)
    {
        localB[i] -= factor * temp[i];
    }
}

int main(int argc, char **argv)
{
    int rank, p;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 4;
    int m = 5; // number of columns (n + 1)

    double A[4][5];        // 4x5 matrix
    double localB[5];      // Row for each process
    double temp[5];        // Temporary row for broadcast
    double reducedA[4][5]; // Reduced matrix after gathering

    if (rank == 0)
    {
        ifstream inputFile("input.txt");
        if (inputFile.is_open())
        {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n + 1; ++j)
                    inputFile >> A[i][j];
            inputFile.close();
        }
        else
        {
            cout << "Unable to open input file" << endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Scatter rows of A to each process
    MPI_Scatter(A, m, MPI_DOUBLE, localB, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initial pivot row setup
    if (rank == 0)
        memcpy(temp, localB, m * sizeof(double));

    for (int k = 0; k < n; k++)
    {
        // Broadcast pivot row to all processes
        MPI_Bcast(temp, m, MPI_DOUBLE, k, MPI_COMM_WORLD);

        if (rank > k)
        {
            gauss(localB, temp, k, m);
        }

        // Update temp to the current row for next pivot broadcast
        if (rank == k && k < n - 1)
            memcpy(temp, localB, m * sizeof(double));
    }

    // Gather the rows back to the root process
    MPI_Gather(localB, m, MPI_DOUBLE, reducedA, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Output the reduced matrix to file and console
        ofstream outputFile("outputfile.txt");
        if (outputFile.is_open())
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n + 1; ++j)
                {
                    outputFile << reducedA[i][j] << "    ";
                    cout << reducedA[i][j] << "    ";
                }
                outputFile << endl;
                cout << endl;
            }
            outputFile << endl;

            // Back substitution
            double ans[n];
            for (int i = n - 1; i >= 0; --i)
            {
                double tmp = reducedA[i][n];
                for (int j = i + 1; j < n; ++j)
                {
                    tmp -= reducedA[i][j] * ans[j];
                }
                ans[i] = tmp / reducedA[i][i];
            }

            // Print solution
            cout << "Resultant values of X are::" << endl;
            for (int i = 0; i < n; ++i)
            {
                outputFile << "x" << i + 1 << ": " << ans[i] << endl;
                cout << "x" << i + 1 << ": " << ans[i] << endl;
            }
            outputFile.close();
        }
        else
        {
            cout << "Unable to open output file" << endl;
        }
    }
    MPI_Finalize();
    return 0;
}
