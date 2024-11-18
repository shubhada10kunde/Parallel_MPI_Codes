if (n % nproc != 0)
{
    if (rank == 0)
    {
        cout << "Matrix size is not divisible by number of processes." << endl;
    }
    MPI_Finalize();
    return 1;
}