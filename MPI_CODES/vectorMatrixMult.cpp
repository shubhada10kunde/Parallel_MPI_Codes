#include <iostream>
#include <vector>
#include <mpi.h>
using namespace std;

int main(int argc, char **argv)
{
    int rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n;
    if (rank == 0)
    {
        cout << "Enter size";
        cin >> n;
    }

    // Broadcast matrix size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_processor = n / nproc;
    if (n % nproc != 0)
    {
        if (rank == 0)
        {
            cout << "Matrix size is not divisible by number of processes." << endl;
        }
        exit(1);
    }

    // Define global matrix and vector on rank 0
    // Defining n*n matrix
    vector<vector<int>> matrix_A(n, vector<int>(n));
    // Defining n*1 vector
    vector<int> vect_X(n);
    vector<int> AnsY_Mat(n);

    // local storage for each processor
    vector<int> local_matrix(n * rows_per_processor);
    vector<int> local_YMat(rows_per_processor, 0); // Defining local y matrix
    vector<int> local_vector(n / nproc);

    if (rank == 0)
    {
        cout << "enter matrix elements" << endl;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                cin >> matrix_A[i][j];
            }
        }

        cout << "matrix is" << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                cout << matrix_A[i][j] << " ";
            }
            cout << endl;
        }

        cout << "enter vector elements" << endl;

        for (int j = 0; j < n; j++)
        {
            cin >> vect_X[j];
        }

        cout << "X vector is" << endl;
        for (int j = 0; j < n; j++)
        {
            cout << vect_X[j] << " ";
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
        MPI_Scatter(temp_matrix.data(), n * rows_per_processor, MPI_INT, local_matrix.data(), n * rows_per_processor, MPI_INT, 0, MPI_COMM_WORLD);

        // Scatter the vector to all processes
        MPI_Scatter(vect_X.data(), n / nproc, MPI_INT, local_vector.data(), n / nproc, MPI_INT, 0, MPI_COMM_WORLD);
    }

    else
    {
        // Scatter matrix rows and vector
        MPI_Scatter(nullptr, 0, MPI_INT, local_matrix.data(), rows_per_processor * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, 0, MPI_INT, local_vector.data(), n / nproc, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // matrix vector multiplication
    for (int i = 0; i < rows_per_processor; i++)
    {
        local_YMat[i] = 0;
        for (int j = 0; j < n; j++)
        {
            local_YMat[i] += local_matrix[i * n + j] * local_vector[j];
        }
    }

    // vector<int> gathered_vector(rows_per_processor);
    // Gather results on root process
    // MPI_Allgather(&local_vector_rows, rows_per_processor, MPI_INT, &AnsY_Mat, rows_per_processor, MPI_INT, MPI_COMM_WORLD);

    // ans vector // gathering results on root process i.e 0th;
    MPI_Gather(local_YMat.data(), rows_per_processor, MPI_INT, AnsY_Mat.data(), rows_per_processor, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Resultant Y vector is:" << endl;
        for (int i = 0; i < n; i++)
        {
            cout << AnsY_Mat[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
