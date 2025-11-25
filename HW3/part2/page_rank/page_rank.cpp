#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>


#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }

    /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */
      // allocate two working arrays: score_old (current) and score_new (next)
  double *score_old = (double *)malloc(sizeof(double) * nnodes);
  double *score_new = (double *)malloc(sizeof(double) * nnodes);
  if (score_old == NULL || score_new == NULL)
  {
    // allocation failed; leave solution as initialized
    if (score_old)
      free(score_old);
    if (score_new)
      free(score_new);
    return;
  }

  // initialize score_old from solution
  for (int i = 0; i < nnodes; ++i)
    score_old[i] = solution[i];

  const double base = (1.0 - damping) / (double)nnodes;

  double global_diff = 0.0;
  const int max_iters = 1000; // safety cap

  for (int iter = 0; iter < max_iters; ++iter)
  {
    // 1) compute dangling mass: sum of scores for nodes with no outgoing edges
    double dangling_sum = 0.0;
    #pragma omp parallel for reduction(+ : dangling_sum) schedule(static)
    for (int v = 0; v < nnodes; ++v)
    {
      if (outgoing_size(g, v) == 0)
        dangling_sum += score_old[v];
    }

    // 2) compute new scores for each node from incoming edges
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nnodes; ++i)
    {
      double sum = 0.0;
      const Vertex *in_b = incoming_begin(g, i);
      const Vertex *in_e = incoming_end(g, i);
      for (const Vertex *it = in_b; it != in_e; ++it)
      {
        Vertex src = *it;
        int out_deg = outgoing_size(g, src);
        if (out_deg > 0)
          sum += score_old[src] / (double)out_deg;
        else
          ; // dangling nodes are handled globally below
      }

      double val = damping * (sum + dangling_sum / (double)nnodes) + base;
      score_new[i] = val;
    }

    // 3) check convergence: compute L1 norm of difference
    global_diff = 0.0;
    #pragma omp parallel for reduction(+ : global_diff) schedule(static)
    for (int i = 0; i < nnodes; ++i)
    {
      global_diff += fabs(score_new[i] - score_old[i]);
    }

    // swap score_old and score_new for next iteration
    double *tmp = score_old;
    score_old = score_new;
    score_new = tmp;

    if (global_diff < convergence)
      break;
  }

  // copy final scores into solution
  for (int i = 0; i < nnodes; ++i)
    solution[i] = score_old[i];

  free(score_old);
  free(score_new);
}
