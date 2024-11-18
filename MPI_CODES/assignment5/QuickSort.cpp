#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <mpi.h>
using namespace std;

// partition the array for quicksort
int partition(vector<int> &arr, int low, int high)
{
    int pivot = arr[high]; // last element as pivot
    int i = (low - 1);     // smaller element

    for (int j = low; j < high; j++)
    {
        if (arr[j] > pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// Quick Sort function
void quickSort(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main(int argc, char **argv)
{
    int rank, nproc;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<int> arr;

    if (rank == 0)
    {
        // Read the array from a text file
        ifstream infile("input.txt");
        string line;

        // if (!infile)
        // {
        //     std::cerr << "Error opening input file!" << std::endl;
        //     MPI_Abort(MPI_COMM_WORLD, 1);
        // }
        // quickSort(arr, 0, n - 1);

        if (!infile)
        {
            cerr << "Error opening input file!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read numbers from the file
        while (getline(infile, line))
        {
            stringstream ss(line);
            int number;
            while (ss >> number)
            {
                arr.push_back(number);
            }
        }
        infile.close();

        // MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Unsorted array: ";
        for (int x : arr)
        {
            cout << x << " ";
        }
        cout << endl;
    }

    int n;
    if (rank == 0)
    {
        n = arr.size();
    }
    // Broadcast the size of the array to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local array size
    int k = n / nproc;
    vector<int> local_arr(k);

    // Scatter the array to all processes
    MPI_Scatter(arr.data(), k, MPI_INT, local_arr.data(), k, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process sorts its local array
    quickSort(local_arr, 0, k - 1);

    // Gather the sorted local array at root process
    MPI_Gather(local_arr.data(), k, MPI_INT, arr.data(), k, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Sorting final bcoz it may not fully sorted
        quickSort(arr, 0, n - 1);

        ofstream outfile("output.txt");
        if (!outfile)
        {
            cerr << "Error opening output file!" << endl;
            exit(1);
        }

        outfile << "Sorted array in descending order: ";
        for (const int &x : arr)
        {
            outfile << x << " ";
        }
        outfile.close();

        cout << "Sorted array has been written to output.txt" << endl;
    }

    MPI_Finalize();
    return 0;
}
