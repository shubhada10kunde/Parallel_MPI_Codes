#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

void GaussianElimination(vector<double> &localB, vector<double> &temporary, int pivot)
{
    double factor = localB[pivot] / temporary[pivot];
    for (int i = 0; i < localB.size(); ++i)
    {
        localB[i] -= factor * temporary[i];
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

    vector<vector<double>> A(n, vector<double>(m)); //  4x5 matrix
    vector<double> localB(m), temp(m);              // Row for each process
    vector<vector<double>> reducedA(n, vector<double>(m));

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

    // Flatten matrix A for MPI_Scatter and Gather
    vector<double> flatA;
    if (rank == 0)
    {
        for (const auto &row : A)
            flatA.insert(flatA.end(), row.begin(), row.end());
    }

    // Scatter rows of A to each process
    MPI_Scatter(flatA.data(), m, MPI_DOUBLE, localB.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0)
        temp = localB;

    for (int k = 1; k < n; k++)
    {
        MPI_Bcast(temp.data(), m, MPI_DOUBLE, k - 1, MPI_COMM_WORLD);
        if (rank > k - 1)
        {
            GaussianElimination(localB, temp, k - 1);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == k && rank != n - 1)
            temp = localB;
    }

    // Gather results back to root process
    vector<double> flatReducedA(n * m);
    MPI_Gather(localB.data(), m, MPI_DOUBLE, flatReducedA.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Convert flatReducedA to 2D vector reducedA
        for (int i = 0; i < n; ++i)
            copy(flatReducedA.begin() + i * m, flatReducedA.begin() + (i + 1) * m, reducedA[i].begin());

        // reduced matrix
        ofstream outputFile("output.txt");
        if (outputFile.is_open())
        {
            // Print the reduced matrix
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

            // back substitution
            vector<double> ans(n);
            for (int i = n - 1; i >= 0; --i)
            {
                double tmp = reducedA[i][n];
                for (int j = i + 1; j < n; ++j)
                {
                    tmp -= reducedA[i][j] * ans[j];
                }
                ans[i] = tmp / reducedA[i][i];
            }

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
        // for (int i = 0; i < n; ++i)
        // {
        //     cout << "x" << i + 1 << ": " << ans[i] << endl;
        // }
    }
    MPI_Finalize();
    return 0;
}
