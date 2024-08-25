#include <bits/stdc++.h>
#include <vector>
#include <mpi.h>
using namespace std;

// function to find f(x)
double function_X(double x)
{
    return sin(x);
}

vector<double> calculatePartitions(int a, int b, int n)
{
    vector<double> Partitions_Array(n + 1);
    // difference between two partitions : h
    double h = (b - a) / static_cast<double>(n); // casting for double
    Partitions_Array[0] = a;                     // 1st partition element
    for (int i = 1; i <= n; i++)
    {
        // for loop for: x0 = a, x1=x0+h, x2=x1+2h, x3=x2+3h, x4=x3+4h
        Partitions_Array[i] = a + i * h;
    }
    cout << "Partition Array:: [";
    for (double partition : Partitions_Array)
    {
        cout << partition << " ";
    }
    cout << "]" << endl;
    return Partitions_Array;
}

// Function to compute the integral
double boolsIntegration(int a, int b, int n, int begin, int to, double h)
{
    vector<double> partitions = calculatePartitions(a, b, n);
    // int sum = 7 * function_X(a);
    int k = (n / 4);
    // double h = (b - a) / static_cast<double>(n);
    double sum = 0.0;
    for (int i = begin; i < k; i++)
    {
        // int xi = a + i * h;
        double x0 = a + i * 4 * h;
        double x1 = x0 + h;
        double x2 = x0 + 2 * h;
        double x3 = x0 + 3 * h;
        double x4 = x0 + 4 * h;

        sum = sum + (7 * function_X(x0) + 32 * function_X(x1) + 12 * function_X(x2) + 32 * function_X(x3) + 7 * function_X(x4));
    }
    return (2 * h / 45.0) * sum; // Apply Boole's Rule formula
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int a = 0, b = 1; // n partitions begin 0 to 1
    int n = 100;      // no of partitions
    int k = (n / 4);
    double h = (double)(b - a) / n; // difference between two partitions

    // PARALLEL PROGRAMMING IMPLEMENTATION

    double local_result = 0.0;
    if (rank == 0)
    {
        // result for part assigned to 0
        int begin = 0;     // start point for process 0
        int to = n / size; // end point // process 0 will calculate integrals begin 0 to 25
        local_result = boolsIntegration(a, b, n, begin, to, h);

        for (int i = 1; i < size; i++)
        {
            // sending each process a part of partition array
            int begin = i * n / size;
            int to = (i + 1) * n / size;
            MPI_Send(&begin, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&to, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // receiving result from all processes other than 0
        double Global_Result = local_result;
        for (int i = 1; i < size; ++i)
        {
            double temp_result;
            MPI_Recv(&temp_result, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Global_Result += temp_result;
        }

        cout << "Booles Rule result::" << Global_Result << endl;
    }
    else
    {
        // receiving data from process other than root process(0) and send results to 0 process
        int begin, to;
        MPI_Recv(&begin, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&to, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        local_result = boolsIntegration(a, b, n, begin, to, h);

        // sending result to process 0
        MPI_Send(&local_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // calculatePartitions(a, b, n);

    MPI_Finalize();
    return 0;
}