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
    // Defining n*n matrix
    vector<vector<int>> matrix_A(n, vector<int>(n));
    // Defining n*1 vector
    vector<int> vect_X(n);
    vector<int> AnsY_Mat(n, 0);

    // local storage for each processor
    vector<int> local_matrix(n * k);
    vector<int> local_vector(k);
    vector<int> local_YMat(k, 0); // Defining local y matrix

    if (rank == 0)
    {
        // Reading matrix and vector from file
        ifstream input_file("matrix.txt");
        if (!input_file)
        {
            cerr << "Error opening input file!" << endl;
            exit(1);
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                input_file >> matrix_A[i][j];
            }
        }

        for (int j = 0; j < n; ++j)
        {
            input_file >> vect_X[j];
        }
        input_file.close();

        // Print matrix and vector (for debugging purposes)
        cout << "Matrix A:" << endl;
        for (const auto &row : matrix_A)
        {
            for (int val : row)
            {
                cout << val << " ";
            }
            cout << endl;
        }

        cout << "Vector X:" << endl;
        for (int val : vect_X)
        {
            cout << val << " ";
        }
        cout << endl;
    }

    // scatter matrix rows to all processors
    vector<int> temp_matrix(n * n);
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp_matrix[i * n + j] = matrix_A[i][j];
            }
        }
    }

    MPI_Scatter(temp_matrix.data(), n * k, MPI_INT, local_matrix.data(), n * k, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the vector to all processes
    MPI_Scatter(vect_X.data(), k, MPI_INT, local_vector.data(), k, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> gathered_vector(n);
    MPI_Allgather(&local_vector, k, MPI_INT, &gathered_vector, k, MPI_INT, MPI_COMM_WORLD);

    // matrix vector multiplication
    for (int i = 0; i < k; i++)
    {
        local_YMat[i] = 0;
        for (int j = 0; j < n; j++)
        {
            local_YMat[i] += local_matrix[i * n + j] * gathered_vector[j];
        }
    }

    // gathering results on root process i.e 0th;
    MPI_Gather(&local_YMat, k, MPI_INT, &AnsY_Mat, k, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Write result to output file
        ofstream output_file("output3.txt");
        if (!output_file)
        {
            cerr << "Error opening output file!" << endl;
            exit(1);
        }

        output_file << "Resultant Y vector:" << endl;
        for (int val : AnsY_Mat)
        {
            output_file << val << " ";
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
