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

    // Exact work distribution across ranks
    long long int base = tosses / world_size;
    long long int rem = tosses % world_size;
    long long int local_tosses = base + (world_rank < rem ? 1 : 0);

    // Per-rank RNG seed
    unsigned int seed = (unsigned int)(time(NULL) * (world_rank + 1));

    // Local Monte Carlo
    long long int local_count = 0;
    for (long long int i = 0; i < local_tosses; ++i) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0) ++local_count;
    }

    // TODO: use MPI_Reduce
    long long int global_count = 0; // receive buffer on root must be distinct from send buffer
    MPI_Reduce(&local_count, &global_count,
               1, MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4.0 * (double)global_count / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}


