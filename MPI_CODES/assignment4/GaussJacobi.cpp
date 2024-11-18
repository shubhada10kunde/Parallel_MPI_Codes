#include <bits/stdc++.h>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <fstream>
using namespace std;

const int MAX_ITER = 1000;
const double TOL = 1e-6;

void print_vector(const vector<double> &vec, ofstream &outfile)
{
    for (const auto &val : vec)
    {
        cout << val << " ";
        outfile << val << " ";
    }
    cout << endl;
    outfile << endl;
}

int main(int argc, char **argv)
{
    int rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n, m;
    vector<double> Mat_A;
    vector<double> Mat_B;
    vector<double> X_oldMat;
    vector<double> Ans;

    if (rank == 0)
    {
        ifstream infile("Inputmatrix.txt");
        if (!infile.is_open())
        {
            cout << "Error opening file!" << endl;
            exit(1);
        }

        // Read dimensions
        infile >> n >> m;
        Mat_A.resize(n * n);
        Mat_B.resize(n);
        X_oldMat.resize(n, 0.0); // Initialize previous solution to zero
        Ans.resize(n);           // Initialize solution vector

        // Read the matrix and vector from the input file
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                infile >> Mat_A[i * n + j];
            }
            infile >> Mat_B[i];
        }

        infile.close();
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize matrices for all processes
    if (rank != 0)
    {
        Mat_A.resize(n * n);
        Mat_B.resize(n);
        X_oldMat.resize(n, 0.0);
        Ans.resize(n);
    }

    // Scatter the matrix rows and B vector
    int k = n / nproc;                // Number of local rows each process
    vector<double> local_AMat(n * k); // Buffer for scattered matrix
    vector<double> local_B(k);        // Buffer for scattered vector
    vector<double> local_Xnew(k, 0.0);

    MPI_Scatter(Mat_A.data(), n * k, MPI_DOUBLE, local_AMat.data(), n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Mat_B.data(), k, MPI_DOUBLE, local_B.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int it = 0; it < MAX_ITER; it++)
    {
        // Jacobi method
        for (int i = 0; i < k; i++)
        {
            double LHS = 0.0;
            int global_i = i + rank * k;
            for (int j = 0; j < n; j++)
            {
                if (global_i != j)
                {
                    LHS += local_AMat[i * n + j] * X_oldMat[j];
                }
            }

            if (local_AMat[i * n + global_i] == 0)
            {
                cout << "Error: Zero found on diagonal at process " << rank << ", row " << global_i << endl;
                exit(1);
            }

            local_Xnew[i] = (local_B[i] - LHS) / local_AMat[i * n + global_i];
        }

        MPI_Allgather(local_Xnew.data(), k, MPI_DOUBLE, Ans.data(), k, MPI_DOUBLE, MPI_COMM_WORLD);

        bool converged = true;
        if (rank == 0)
        {
            for (int i = 0; i < n; i++)
            {
                if (fabs(X_oldMat[i] - Ans[i]) >= TOL)
                {
                    converged = false;
                    break;
                }
            }
            if (converged)
            {
                break;
            }
        }

        X_oldMat = Ans;
    }

    if (rank == 0)
    {
        ofstream outfile("solution_output.txt");
        if (!outfile.is_open())
        {
            cout << "Error opening output file!" << endl;
            exit(1);
        }

        outfile << "Final Solution:" << endl;
        print_vector(Ans, outfile);
        outfile.close();
    }

    MPI_Finalize();
    return 0;
}
