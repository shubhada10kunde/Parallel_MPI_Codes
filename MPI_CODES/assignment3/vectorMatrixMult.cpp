#include <iostream>
#include <vector>
#include <mpi.h>
#include <fstream>
using namespace std;

int main(int argc, char **argv)
{
    int rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 4;
    int k = n / nproc;

    // Define global matrix and vector on rank 0
    vector<int> matrix_A(n * n); // Defining n*n matrix
    vector<int> vect_X(n);       // Defining n*1 vector
    vector<int> AnsY_Mat(n);     // Gathered results

    // local storage for each processor
    vector<int> local_matrix(n * k); // Buffer for scattered matrix
    vector<int> local_vector(k);     // Buffer for scattered vector
    vector<int> local_YMat(k);       // Result of matrix-vector multiplication

    if (rank == 0)
    {
        // Reading matrix and vector from file
        ifstream input_file("Inputmatrix.txt");
        if (!input_file)
        {
            cerr << "Error opening input file!" << endl;
            exit(1);
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                input_file >> matrix_A[i * n + j];
            }
        }

        for (int j = 0; j < n; ++j)
        {
            input_file >> vect_X[j];
        }
        input_file.close();

        cout << "Matrix A:" << endl;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                cout << matrix_A[i * n + j] << " ";
            }
            cout << endl;
        }

        cout << "Vector X:" << endl;
        for (int j = 0; j < n; ++j)
        {
            cout << vect_X[j] << " ";
        }
        cout << endl;
    }

    // scatter matrix rows to all processors
    MPI_Scatter(matrix_A.data(), n * k, MPI_INT, local_matrix.data(), n * k, MPI_INT, 0, MPI_COMM_WORLD);
    // cout << rank << endl;
    // for (int i = 0; i < n * k; i++)
    // {
    //     cout << local_matrix[i];
    // }
    // cout << endl;

    // Scatter the vector to all processes
    MPI_Scatter(vect_X.data(), k, MPI_INT, local_vector.data(), k, MPI_INT, 0, MPI_COMM_WORLD);
    // cout << rank << endl;
    // for (int i = 0; i < k; i++)
    // {
    //     cout << local_vector[i];
    // }
    // cout << endl;

    vector<int> gathered_vector(n);
    MPI_Allgather(local_vector.data(), k, MPI_INT, gathered_vector.data(), k, MPI_INT, MPI_COMM_WORLD);
    // cout << rank << endl;
    // for (int i = 0; i < k; i++)
    // {
    //     cout << gathered_vector[i];
    // }
    // cout << endl;

    // // matrix vector multiplication
    for (int i = 0; i < k; i++)
    {
        local_YMat[i] = 0;
        for (int j = 0; j < n; j++)
        {
            local_YMat[i] += local_matrix[i * n + j] * gathered_vector[j];
        }
    }

    // gathering results on root process i.e 0th;
    MPI_Gather(local_YMat.data(), k, MPI_INT, AnsY_Mat.data(), k, MPI_INT, 0, MPI_COMM_WORLD);
    cout << rank << endl;
    for (int i = 0; i < k; i++)
    {
        cout << AnsY_Mat[i];
    }
    cout << endl;

    if (rank == 0)
    {
        // Write result to output file
        ofstream output_file("output3.txt");
        if (!output_file)
        {
            cerr << "Error opening output file!" << endl;
            exit(1);
        }

        output_file << "Resultant answer vector:" << endl;
        for (int i = 0; i < n; ++i)
        {
            output_file << AnsY_Mat[i] << endl;
        }
        output_file << endl;
        output_file.close();

        for (int i = 0; i < n; ++i)
        {
            cout << "Resultant Y vector:" << i << " is " << AnsY_Mat[i] << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
