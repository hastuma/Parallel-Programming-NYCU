#include "bfs.h"
#include <cstdlib>
#include <vector>
#include <omp.h>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE


constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    int frontier_count = frontier->count;

    int max_threads = omp_get_max_threads();
    std::vector<int> counts(max_threads);
    std::vector<int> offsets(max_threads);

    #pragma omp parallel default(none) shared(g, frontier, new_frontier, distances, frontier_count, counts, offsets, max_threads)
    {
        int tid = omp_get_thread_num();
        std::vector<int> local_buf;
        local_buf.reserve(256);

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < frontier_count; i++)
        {
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // add  neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];
                if (distances[outgoing] != NOT_VISITED_MARKER)
                    continue;
                int expected = NOT_VISITED_MARKER;
                int new_dist = distances[node] + 1;
                if (__sync_bool_compare_and_swap(&distances[outgoing], expected, new_dist))
                {
                    local_buf.push_back(outgoing);
                }
            }
        }

        // local count
        counts[tid] = (int)local_buf.size();
        #pragma omp barrier
        #pragma omp single
        {
            int nthreads = omp_get_num_threads();
            int sum = 0;
            for (int t = 0; t < nthreads; ++t)
            {
                offsets[t] = sum;
                sum += counts[t];
            }
            new_frontier->count = sum;
        }

        #pragma omp barrier
        // copy local buffer into global frontier
        int my_offset = offsets[tid];
        for (int k = 0; k < (int)local_buf.size(); ++k)
            new_frontier->vertices[my_offset + k] = local_buf[k];
    }
}

void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize 
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap ptr
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize 
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int cur_level = 0;

    // alloc reusable per-thread  arrays 
    int max_threads = omp_get_max_threads();
    std::vector<int> counts(max_threads);
    std::vector<int> offsets(max_threads);
    std::vector<unsigned char> frontier_bitmap(graph->num_nodes);

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);
        int n = graph->num_nodes;
        // set bits for nodes in frontier
        for (int i = 0; i < frontier->count; ++i)
            frontier_bitmap[frontier->vertices[i]] = 1;

        int est = std::max(16, frontier->count / (max_threads == 0 ? 1 : max_threads));
        #pragma omp parallel default(none) shared(graph, sol, cur_level, n, counts, offsets, new_frontier, frontier_bitmap, est)
        {
            int tid = omp_get_thread_num();
            std::vector<int> local_buf;
            // reserve using estimated per-thread size
            local_buf.reserve(est);
            #pragma omp for schedule(dynamic, 1024)
            for (int v = 0; v < n; ++v)
            {
                // skip already visited
                if (sol->distances[v] != NOT_VISITED_MARKER)
                    continue;
                // check incoming neighbors for membership in frontier using bitmap
                const Vertex *in_b = incoming_begin(graph, v);
                const Vertex *in_e = incoming_end(graph, v);
                for (const Vertex *it = in_b; it != in_e; ++it)
                {
                    Vertex u = *it;
                    if (frontier_bitmap[u])
                    {
                        // attempt to claim v
                        int expected = NOT_VISITED_MARKER;
                        int new_dist = cur_level + 1;
                        if (__sync_bool_compare_and_swap(&sol->distances[v], expected, new_dist))
                        {
                            local_buf.push_back(v);
                        }
                        break;
                    }
                }
            }

            counts[tid] = (int)local_buf.size();

            #pragma omp barrier
            #pragma omp single
            {
                int nth = omp_get_num_threads();
                int sum = 0;
                for (int t = 0; t < nth; ++t)
                {
                    offsets[t] = sum;
                    sum += counts[t];
                }
                new_frontier->count = sum;
            }

            #pragma omp barrier
            int my_offset = offsets[tid];
            for (int k = 0; k < (int)local_buf.size(); ++k)
                new_frontier->vertices[my_offset + k] = local_buf[k];
        }

        // clear frontier bitmap 
        for (int i = 0; i < frontier->count; ++i)
            frontier_bitmap[frontier->vertices[i]] = 0;

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap and advance level
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        cur_level++;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initial
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int cur_level = 0;

    const double avg_deg = (double)graph->num_edges / (double)graph->num_nodes;

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        // estimate top-down work vs bottom-up work
        double est_td = (double)frontier->count * avg_deg;
        bool use_top_down = (est_td <= (double)graph->num_nodes);

        if (use_top_down)
        {
            //  top-down step
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }
        else
        {
            // bottom-up step 
            int n = graph->num_nodes;
            int max_threads = omp_get_max_threads();
            std::vector<int> counts(max_threads);
            std::vector<int> offsets(max_threads);

            #pragma omp parallel default(none) shared(graph, sol, cur_level, n, counts, offsets, new_frontier)
            {
                int tid = omp_get_thread_num();
                std::vector<int> local_buf;
                local_buf.reserve(256);

                #pragma omp for schedule(static)
                for (int v = 0; v < n; ++v)
                {
                    if (sol->distances[v] != NOT_VISITED_MARKER)
                        continue;

                    const Vertex *in_b = incoming_begin(graph, v);
                    const Vertex *in_e = incoming_end(graph, v);
                    for (const Vertex *it = in_b; it != in_e; ++it)
                    {
                        Vertex u = *it;
                        if (sol->distances[u] == cur_level)
                        {
                            int expected = NOT_VISITED_MARKER;
                            int new_dist = cur_level + 1;
                            if (__sync_bool_compare_and_swap(&sol->distances[v], expected, new_dist))
                            {
                                local_buf.push_back(v);
                            }
                            break;
                        }
                    }
                }

                counts[tid] = (int)local_buf.size();

                #pragma omp barrier
                #pragma omp single
                {
                    int nth = omp_get_num_threads();
                    int sum = 0;
                    for (int t = 0; t < nth; ++t)
                    {
                        offsets[t] = sum;
                        sum += counts[t];
                    }
                    new_frontier->count = sum;
                }

                #pragma omp barrier
                int my_offset = offsets[tid];
                for (int k = 0; k < (int)local_buf.size(); ++k)
                    new_frontier->vertices[my_offset + k] = local_buf[k];
            }
        }

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap and advance
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        cur_level++;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}






