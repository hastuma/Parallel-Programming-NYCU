#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>


static inline uint64_t rotl(const uint64_t x, int k) 
{
    return (x << k) | (x >> (64 - k));
}

struct xoshiro256pp_state
{
    uint64_t s[4];
};

static inline uint64_t xoshiro256pp_next(struct xoshiro256pp_state *state) 
{
    const uint64_t result = rotl(state->s[0] + state->s[3], 23) + state->s[0];
    const uint64_t t = state->s[1] << 17;
    
    state->s[2] ^= state->s[0];
    state->s[3] ^= state->s[1];
    state->s[1] ^= state->s[2];
    state->s[0] ^= state->s[3];
    state->s[2] ^= t;
    state->s[3] = rotl(state->s[3], 45);
    
    return result;
}

// Convert uint64 to double in [-1, 1]
static inline double to_double_pm1(uint64_t x) {
    // Map to [0, 1) then scale to [-1, 1)
    return ((x >> 11) * 0x1.0p-53) * 2.0 - 1.0;
}

struct thread_data {
    int thread_id;
    long long tosses;
    long long hits;
    struct xoshiro256pp_state rng; // 每個threads 自己都有一個ＲＮＧ就不會race
};

void* monte_carlo_worker(void* arg) {
    struct thread_data* data = (struct thread_data*)arg;
    struct xoshiro256pp_state* rng = &data->rng;
    
    long long local_hits = 0;
    long long tosses = data->tosses;
    
    // Main Monte Carlo loop - optimized for speed
    for (long long i = 0; i < tosses; ++i) 
    {
        double x = to_double_pm1(xoshiro256pp_next(rng));
        double y = to_double_pm1(xoshiro256pp_next(rng));
        if (x * x + y * y <= 1.0) 
        {
            ++local_hits;
        }
    }
    data->hits = local_hits;
    return NULL;
}


int main(int argc, char* argv[]) 
{
    if (argc != 3) 
    {
        fprintf(stderr, "Usage: %s <num_threads> <num_tosses>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    long long num_tosses = atoll(argv[2]);
    
    if (num_threads <= 0 || num_tosses <= 0) 
    {
        fprintf(stderr, "Error: Invalid arguments\n");
        return 1;
    }
    
    // Limit threads to number of tosses if needed
    if (num_threads > num_tosses) 
    {
        num_threads = (int)num_tosses;
    }
    
    // Allocate thread data and handles
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    struct thread_data* tdata = (struct thread_data*)malloc(num_threads * sizeof(struct thread_data));
    
    // Initialize RNG seeds using time and thread id
    uint64_t base_seed = (uint64_t)time(NULL);
    
    // Distribute work among threads
    long long base_tosses = num_tosses / num_threads;
    long long remainder = num_tosses % num_threads;
    
    // Create threads
    for (int i = 0; i < num_threads; ++i) {
        tdata[i].thread_id = i;
        tdata[i].tosses = base_tosses + (i < remainder ? 1 : 0);
        tdata[i].hits = 0;
        
        // Initialize RNG state for this thread (different seed per thread)
        uint64_t seed = base_seed + i * 0x9e3779b97f4a7c15ULL;
        tdata[i].rng.s[0] = seed;
        tdata[i].rng.s[1] = seed ^ 0x1234567896969699ULL;
        tdata[i].rng.s[2] = seed ^ 0xf666666666643210ULL;
        tdata[i].rng.s[3] = seed ^ 0x0246813579bdf024ULL;
        pthread_create(&threads[i], NULL, monte_carlo_worker, &tdata[i]);
    }
    
    
    long long total_hits = 0;
    for (int i = 0; i < num_threads; ++i) 
    {
        pthread_join(threads[i], NULL);
        total_hits += tdata[i].hits;
    }
    
    // Calculate PI estimate
    double pi_estimate = 4.0 * (double)total_hits / (double)num_tosses;
    
    // Print result (as required by spec: print the estimate of PI)
    printf("%.10f\n", pi_estimate);
    
    // Cleanup
    free(threads);
    free(tdata);
    
    return 0;
}