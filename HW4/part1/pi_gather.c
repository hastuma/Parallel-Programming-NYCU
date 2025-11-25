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
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Distribute iterations per process exactly
    long long int base = tosses / world_size;
    long long int rem = tosses % world_size;
    long long int local_tosses = base + (world_rank < rem ? 1 : 0);

    // Per-rank RNG seed so each process has different random sequence
    unsigned int seed = (unsigned int)(time(NULL) * (world_rank + 1));

    long long int local_count = 0;
    for (long long int i = 0; i < local_tosses; ++i) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            ++local_count;
    }

    // TODO: use MPI_Gather
    long long int *counts = NULL;
    if (world_rank == 0) {
        counts = (long long int*)malloc(sizeof(long long int) * world_size);
    } 
    MPI_Gather(&local_count, 1, MPI_LONG_LONG,
               counts,      1, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        long long int total_count = 0;
        for (int i = 0; i < world_size; ++i) total_count += counts[i];
        pi_result = 4.0 * (double)total_count / (double)tosses;
        free(counts);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
