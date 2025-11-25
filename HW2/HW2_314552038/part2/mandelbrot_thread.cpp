#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>  // Use standard C++ timing instead

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{
    // Start timing for this thread
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Use interleaved row assignment for better load balancing
    // Thread i processes rows: i, i+numThreads, i+2*numThreads, ...
    // This ensures each thread gets a mix of simple and complex rows
    
    int total_rows = args->height;
    int thread_id = args->threadId;
    int num_threads = args->numThreads;
    
    // Process rows in an interleaved pattern
    for (int row = thread_id; row < total_rows; row += num_threads)
    {
        // Process one row at a time
        mandelbrot_serial(args->x0,
                         args->y0,
                         args->x1,
                         args->y1,
                         args->width,
                         args->height,
                         row,        // start_row
                         1,          // num_rows (process 1 row at a time)
                         args->maxIterations,
                         args->output);
    }
    
    // End timing for this thread
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = endTime - startTime;
    
    printf("[Thread %d]:\t\t[%.3f] ms\n", thread_id, elapsed.count());
}

//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = max_iterations;
        args[i].numThreads = num_threads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Print header for thread timing
    printf("\n--- Thread Timing Breakdown ---\n");

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }
    
    printf("-------------------------------\n\n");
}
