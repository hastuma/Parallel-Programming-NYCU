#include <mpi.h>
#include <cstring>

static inline int rows_for_rank(int n, int size, int rank) {
    int base = n / size;
    int rem = n % size;
    return base + (rank < rem ? 1 : 0);
}

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /* TODO: The data is stored in a_mat and b_mat.
     * You need to allocate memory for a_mat_ptr and b_mat_ptr,
     * and copy the data from a_mat and b_mat to a_mat_ptr and b_mat_ptr, respectively.
     * You can use any size and layout you want if they provide better performance.
     * Unambitiously copying the data is also acceptable.
     *
     * The matrix multiplication will be performed on a_mat_ptr and b_mat_ptr.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Distribute rows of A across ranks; Broadcast full B to all ranks (column-major as given)
    int local_rows = rows_for_rank(n, size, rank);
    int local_a_elems = local_rows * m;

    // Allocate local A buffer (row-major, local_rows x m)
    int *local_A = nullptr;
    if (local_a_elems > 0) local_A = new int[local_a_elems];

    // Root prepares counts and displacements for Scatterv
    int *sendcounts = nullptr;
    int *displs = nullptr;
    if (rank == 0) {
        sendcounts = new int[size];
        displs = new int[size];
        int offset_rows = 0;
        for (int r = 0; r < size; ++r) {
            int r_rows = rows_for_rank(n, size, r);
            sendcounts[r] = r_rows * m; // number of ints
            displs[r] = offset_rows * m; // offset in ints
            offset_rows += r_rows;
        }
    }

    // Scatter rows of A
    MPI_Scatterv(
        a_mat, sendcounts, displs, MPI_INT,
        local_A, local_a_elems, MPI_INT,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        delete[] sendcounts;
        delete[] displs;
    }

    // Allocate B buffer on each rank (store as m x l, column-major like input b_mat)
    int *B = nullptr;
    long long total_b_elems = 1LL * m * l;
    if (total_b_elems > 0) B = new int[m * l];

    if (rank == 0) {
        // Copy root's B into buffer for broadcast
        if (total_b_elems > 0) std::memcpy(B, b_mat, sizeof(int) * (size_t)(m * l));
    }
    // Broadcast full B to all ranks
    MPI_Bcast(B, m * l, MPI_INT, 0, MPI_COMM_WORLD);

    *a_mat_ptr = local_A;
    *b_mat_ptr = B;
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    /* TODO: Perform matrix multiplication on a_mat and b_mat. Which are the matrices you've
     * constructed. The result should be stored in out_mat, which is a continuous memory placing n *
     * l elements of int. You need to make sure rank 0 receives the result.
     */
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Local rows owned by this rank
    int local_rows = rows_for_rank(n, size, rank);

    // Allocate local C buffer (local_rows x l)
    int *local_C = nullptr;
    if (local_rows * l > 0) local_C = new int[local_rows * l];

    // Compute C_local = A_local (local_rows x m, row-major) * B (m x l, column-major)
    // Access b_mat with index b_mat[j*m + k]
    // Simple triple loop; cache-friendly due to contiguous access on k for both A row and B column
    for (int i = 0; i < local_rows; ++i) {
        int *c_row = local_C + i * l;
        // Initialize row to zero
        for (int j = 0; j < l; ++j) c_row[j] = 0;
        const int *a_row = a_mat + i * m;
        for (int k = 0; k < m; ++k) {
            int a_ik = a_row[k];
            const int *b_col_k = b_mat + k; // start of row k in column-major? Actually column j index is j*m + k
            // Accumulate into all columns j
            for (int j = 0; j < l; ++j) {
                // b(k,j) is at b_mat[j*m + k]
                c_row[j] += a_ik * b_mat[j * m + k];
            }
        }
    }

    // Gather local C rows to rank 0 using Gatherv
    int *recvcounts = nullptr;
    int *displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        int offset_rows = 0;
        for (int r = 0; r < size; ++r) {
            int r_rows = rows_for_rank(n, size, r);
            recvcounts[r] = r_rows * l;
            displs[r] = offset_rows * l;
            offset_rows += r_rows;
        }
    }

    MPI_Gatherv(local_C, local_rows * l, MPI_INT,
                out_mat, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        delete[] recvcounts;
        delete[] displs;
    }

    delete[] local_C;
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    // Free the per-rank allocated buffers
    delete[] a_mat;
    delete[] b_mat;
}
