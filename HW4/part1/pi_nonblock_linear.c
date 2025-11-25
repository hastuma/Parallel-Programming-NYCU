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

    // Distribute tosses exactly (first 'rem' ranks get one extra)
    long long int base = tosses / world_size;
    long long int rem = tosses % world_size;
    long long int local_tosses = base + (world_rank < rem ? 1 : 0);

    // Per-rank seed to diversify random streams
    unsigned int seed = (unsigned int)(time(NULL) * (world_rank + 1));

    long long int local_count = 0;
    for (long long int i = 0; i < local_tosses; ++i) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / (double)RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            ++local_count;
    }

    if (world_rank > 0)
    {
        // TODO: MPI workers
        // Linear non-blocking reduction: workers just send their local counts
        MPI_Send(&local_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.

        // MPI_Request requests[];

        // MPI_Waitall();
        long long int total_count = local_count; // include rank 0's own count
        if (world_size > 1) {
            int recv_n = world_size - 1;
            long long int *recv_buf = (long long int*)malloc(sizeof(long long int) * recv_n);
            MPI_Request *requests = (MPI_Request*)malloc(sizeof(MPI_Request) * recv_n);
            if (!recv_buf || !requests) {
                fprintf(stderr, "Allocation failed\n");
            } else {
                // Post all non-blocking receives first (do NOT wait immediately)
                for (int i = 0; i < recv_n; ++i) {
                    int src = i + 1; // ranks 1..world_size-1
                    MPI_Irecv(&recv_buf[i], 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, &requests[i]);
                }
                // Wait for all receives to complete
                MPI_Waitall(recv_n, requests, MPI_STATUSES_IGNORE);
                // Accumulate
                for (int i = 0; i < recv_n; ++i) {
                    total_count += recv_buf[i];
                }
            }
            free(recv_buf);
            free(requests);
        }
        // Store result for printing below
        pi_result = 4.0 * (double)total_count / (double)tosses;
    }

    if (world_rank == 0)
    {
        // TODO: PI result

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
