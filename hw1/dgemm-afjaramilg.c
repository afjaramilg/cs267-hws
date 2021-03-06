#include <immintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define B_BLOCK_ROWS 32
#define B_BLOCK_COLS 32

const char* dgemm_desc =
    "dgemm using blocked algorithm, but with loops in different order";

// static double B_BUFFER[B_BLOCK_COLS * B_BLOCK_ROWS];
// static int B_BUFFER_ROWS, B_BUFFER_COLS;

static inline double m256d_horizontal_add(__m256d a) {
    __m256d t1 = _mm256_hadd_pd(a, a);
    __m128d t2 = _mm256_extractf128_pd(t1, 1);
    __m128d t3 = _mm_add_sd(_mm256_castpd256_pd128(t1), t2);
    return _mm_cvtsd_f64(t3);
}

// normal 1-op-at-a-time block multiply where A, B and C are all column
// major
static void do_block_fixed(const int lda, const double* A, const double* B,
                           double* __restrict__ C) {
    __m256d v1, v2, res[4], total;

    for (int col_b = 0; col_b < B_BLOCK_COLS; ++col_b) {
        for (int row_a = 0; row_a < B_BLOCK_COLS; ++row_a) {
            for (int k = 0; k < B_BLOCK_ROWS; k += 16) {
                // double b_block_val = B_BUFFER[(col_b * B_BLOCK_ROWS) +
                // row_b];
                // double b_block_val = B[(col_b * lda) + k];

                double* START_A = A + (row_a * lda) + k;
                double* START_B = B + (col_b * lda) + k;

                v1 = _mm256_loadu_pd(START_A);
                v2 = _mm256_loadu_pd(START_B);
                res[0] = _mm256_mul_pd(v1, v2);

                v1 = _mm256_loadu_pd(START_A + 4);
                v2 = _mm256_loadu_pd(START_B + 4);
                res[1] = _mm256_mul_pd(v1, v2);

                v1 = _mm256_loadu_pd(START_A + 8);
                v2 = _mm256_loadu_pd(START_B + 8);
                res[2] = _mm256_mul_pd(v1, v2);


                v1 = _mm256_loadu_pd(START_A + 12);
                v2 = _mm256_loadu_pd(START_B + 12);
                res[3] = _mm256_mul_pd(v1, v2);

                total = _mm256_add_pd(res[0], res[1]);
                total = _mm256_add_pd(total, res[2]);
                total = _mm256_add_pd(total, res[3]);


                C[(col_b * lda) + row_a] += m256d_horizontal_add(total);
            }
        }
    }
}

static void do_block(const int lda, const double* A, const double* B,
                     double* __restrict__ C, const int rows_a, const int cols_b,
                     const int max_k) {
    for (int col_b = 0; col_b < cols_b; ++col_b) {
        for (int row_a = 0; row_a < rows_a; ++row_a) {
            for (int k = 0; k < max_k; ++k) {
                // double b_block_val = B_BUFFER[(col_b * B_BUFFER_ROWS) +
                // row_b];
                // double b_block_val = B[(col_b * lda) + k];

                C[(col_b * lda) + row_a] +=
                    A[(row_a * lda) + k] * B[(col_b * lda) + k];
            }
        }
    }
}

static void inv_mat_inplace(const int lda, double* __restrict__ S) {
    for (int col = 0; col < lda; ++col)
        for (int row = col + 1; row < lda; ++row) {
            double cpy = S[(row * lda) + col];
            S[(row * lda) + col] = S[(col * lda) + row];
            S[(col * lda) + row] = cpy;
        }
}

void square_dgemm(const int lda, const double* A, const double* B,
                  double* __restrict__ C) {
    inv_mat_inplace(lda, A);

    for (int col_b = 0; col_b < lda; col_b += B_BLOCK_COLS) {
        int cols_b = min(B_BLOCK_COLS, lda - col_b);

        for (int row_a = 0; row_a < lda; row_a += B_BLOCK_COLS) {
            int rows_a = min(B_BLOCK_COLS, lda - row_a);

            for (int k = 0; k < lda; k += B_BLOCK_ROWS) {
                int max_k = min(B_BLOCK_ROWS, lda - k);

                const double* B_CPY = B + ((col_b * lda) + k);

                // for (int cb = 0; cb < B_BUFFER_COLS; ++cb) {
                // for (int rb = 0; rb < B_BUFFER_ROWS; ++rb) {
                // B_BUFFER[(cb * B_BUFFER_ROWS) + rb] =
                // B_CPY[(cb * lda) + rb];
                //}
                //}
                // printf("b:%i,%i  a:%i,%i\n", row_b, col_b, row_a, row_b);

                if (max_k == B_BLOCK_ROWS && cols_b == B_BLOCK_COLS &&
                    rows_a == B_BLOCK_COLS)
                    do_block_fixed(lda, A + ((row_a * lda) + k), B_CPY,
                                   C + ((col_b * lda) + row_a));
                else
                    do_block(lda, A + ((row_a * lda) + k), B_CPY,
                             C + ((col_b * lda) + row_a), rows_a, cols_b,
                             max_k);
            }
        }
    }

    inv_mat_inplace(lda, A);
    // puts("----------------------------");
}
