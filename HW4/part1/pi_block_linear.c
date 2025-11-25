#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoll(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Calculate iterations per process (not total iterations)
    long long int local_tosses = (tosses + world_size - 1) / world_size;
    
    // Multiply seed with rank to ensure different random numbers per process
    unsigned int seed = (unsigned int)(time(NULL) * (world_rank + 1));
    
    long long int local_count = 0;
    
    for (long long int i = 0; i < local_tosses; ++i)
    {
        double x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            ++local_count;
    }

    if (world_rank > 0)
    {
        MPI_Send(&local_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        long long int total_count = local_count;
        for (int src = 1; src < world_size; ++src)
        {
            long long int recv_count = 0;
            MPI_Recv(&recv_count, 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_count += recv_count;
        }

        pi_result = 4.0 * (double)total_count / (double)tosses;
    }

    if (world_rank == 0)
    {
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
