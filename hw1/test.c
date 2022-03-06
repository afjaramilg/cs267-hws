#include <immintrin.h>
#include <stdio.h>
/*=============================================================================
*******************************************************************************

TRANSPOSE A

*******************************************************************************
===============================================================================
*/

/*
#include <immintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define B_BLOCK_ROWS 32
#define B_BLOCK_COLS 24

const char* dgemm_desc =
    "dgemm using blocked algorithm, but with loops in different order";

static double B_BUFFER[B_BLOCK_COLS * B_BLOCK_ROWS];
static int B_BUFFER_ROWS, B_BUFFER_COLS;

// normal 1-op-at-a-time block multiply where A, B and C are all column
// major
static void do_block_fixed(const int lda, const double* A, const double* B,
                           double* __restrict__ C) {
    for (int col_b = 0; col_b < B_BLOCK_COLS; ++col_b) {
        for (int row_a = 0; row_a < B_BLOCK_COLS; ++row_a) {
            for (int k = 0; k < B_BLOCK_ROWS; ++k) {
                // double b_block_val = B_BUFFER[(col_b * B_BLOCK_ROWS) +
                // row_b];
                // double b_block_val = B[(col_b * lda) + k];

                C[(col_b * lda) + row_a] +=
                    A[(row_a * lda) + k] * B[(col_b * lda) + k];
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


 */

/*=============================================================================
*******************************************************************************

BLOCKED B WITHOUT ANY MEMORY MOVING STUFF

*******************************************************************************
===============================================================================
*/

/*
#include <immintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define B_BLOCK_ROWS 32
#define B_BLOCK_COLS 32

const char* dgemm_desc =
    "dgemm using blocked algorithm, but with loops in different order";

static double B_BUFFER[B_BLOCK_COLS * B_BLOCK_ROWS];
static int B_BUFFER_ROWS, B_BUFFER_COLS;

// normal 1-op-at-a-time block multiply where A, B and C are all column
// major
static void do_block_fixed(const int lda, const double* A, const double* B,
                           double* __restrict__ C) {
    for (int col_b = 0; col_b < B_BLOCK_COLS; ++col_b) {
        for (int row_b = 0; row_b < B_BLOCK_ROWS; ++row_b) {
            // double b_block_val = B_BUFFER[(col_b * B_BLOCK_ROWS) + row_b];
            double b_block_val = B[(col_b * lda) + row_b];

            for (int row_a = 0; row_a < B_BLOCK_ROWS; ++row_a) {
                C[(col_b * lda) + row_a] +=
                    A[(row_b * lda) + row_a] * b_block_val;
            }
        }
    }
}

static void do_block(const int lda, const double* A, const double* B,
                     double* __restrict__ C, const int rows_ba) {
    for (int col_b = 0; col_b < B_BUFFER_COLS; ++col_b) {
        for (int row_b = 0; row_b < B_BUFFER_ROWS; ++row_b) {
            // double b_block_val = B_BUFFER[(col_b * B_BUFFER_ROWS) + row_b];
            double b_block_val = B[(col_b * lda) + row_b];

            for (int row_a = 0; row_a < rows_ba; ++row_a) {
                C[(col_b * lda) + row_a] +=
                    A[(row_b * lda) + row_a] * b_block_val;
            }
        }
    }
}

void square_dgemm(const int lda, const double* A, const double* B,
                  double* __restrict__ C) {
    for (int col_b = 0; col_b < lda; col_b += B_BLOCK_COLS) {
        B_BUFFER_COLS = min(B_BLOCK_COLS, lda - col_b);

        for (int row_b = 0; row_b < lda; row_b += B_BLOCK_ROWS) {
            B_BUFFER_ROWS = min(B_BLOCK_ROWS, lda - row_b);

            const double* B_CPY = B + ((col_b * lda) + row_b);


            //for (int cb = 0; cb < B_BUFFER_COLS; ++cb) {
                //for (int rb = 0; rb < B_BUFFER_ROWS; ++rb) {
                    //B_BUFFER[(cb * B_BUFFER_ROWS) + rb] =
                        //B_CPY[(cb * lda) + rb];
                //}
            //}

            for (int row_a = 0; row_a < lda; row_a += B_BLOCK_COLS) {
                int rows_ba = min(B_BLOCK_COLS, lda - row_a);

                // printf("b:%i,%i  a:%i,%i\n", row_b, col_b, row_a, row_b);

                if (B_BUFFER_ROWS == B_BLOCK_ROWS &&
                    B_BUFFER_COLS == B_BLOCK_COLS && rows_ba == B_BLOCK_COLS)
                    do_block_fixed(lda, A + ((row_b * lda) + row_a), B_CPY,
                                   C + ((col_b * lda) + row_a));
                else
                    do_block(lda, A + ((row_b * lda) + row_a), B_CPY,
                             C + ((col_b * lda) + row_a), rows_ba);
            }
        }
    }

    // puts("----------------------------");
}
 */

/*=============================================================================
*******************************************************************************

 UNFINISHED UNROLLED LAYOUT IDEA THINGAMGGIG

*******************************************************************************
===============================================================================
*/

/*
#include <immintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define B_BLOCK_ROWS (8)
#define B_BLOCK_COLS (8)
#define BLOCK_SIZE (B_BLOCK_ROWS * B_BLOCK_COLS)

const char* dgemm_desc =
    "dgemm using layout change for blocks";

static void unroll_rd(const int lda, const double* __restrict__ S,
                      double* __restrict__ D) {
    const int BLOCK_COL_SIZE = lda * B_BLOCK_COLS;

    int c_offset = 0;
    for (int c = 0; c < (lda / B_BLOCK_COLS) * B_BLOCK_COLS; ++c) {
        const int BLOCK_COL_IND = (c / B_BLOCK_COLS) * BLOCK_COL_SIZE;
        for (int r = 0; r < (lda / B_BLOCK_ROWS) * B_BLOCK_ROWS; ++r) {
            int ind = BLOCK_COL_IND + (r * B_BLOCK_COLS) + c_offset;
            D[ind] = S[(c * lda) + r];
        }

        c_offset = (c_offset + 1) * (c_offset < (B_BLOCK_COLS - 1));
    }
}

static void unroll_dr(const int lda, const double* __restrict__ S,
                      double* __restrict__ D) {
    const int BLOCK_COL_SIZE = lda * B_BLOCK_COLS;

    int r_offset = 0;
    for (int c = 0; c < (lda / B_BLOCK_COLS) * B_BLOCK_COLS; ++c) {
        const int BLOCK_COL_IND = ((c / B_BLOCK_COLS) * BLOCK_COL_SIZE) +
                                  ((c % B_BLOCK_COLS) * B_BLOCK_ROWS);

        for (int r = 0; r < (lda / B_BLOCK_ROWS) * B_BLOCK_ROWS; ++r) {
            int ind =
                BLOCK_COL_IND + ((r / B_BLOCK_ROWS) * BLOCK_SIZE) + r_offset;

            D[ind] = S[(c * lda) + r];
            r_offset = (r_offset + 1) * (r_offset < (B_BLOCK_ROWS - 1));
        }
    }
}

void square_dgemm(const int lda, const double* A, const double* B,
                  double* __restrict__ C) {
    double* ARD = malloc(2 * lda * lda * sizeof(double));
    double* BDR = ARD + (lda * lda);

    unroll_rd(lda, A, ARD);
    unroll_dr(lda, A, BDR);

    //memset(C, 0, lda * lda * sizeof(double));

    const int B_BLOCK_COL_SIZE = lda * B_BLOCK_COLS;
    const int A_BLOCK_COL_SIZE = lda * B_BLOCK_ROWS;

    int count = 0;

    for (int bi = 0; bi < (lda * lda); bi += B_BLOCK_ROWS) {
        const int c_col = ((bi % BLOCK_SIZE) / B_BLOCK_ROWS) +
                          ((bi / B_BLOCK_COL_SIZE) * B_BLOCK_COLS);

        const int a_offset =  // b block row * a block col size
            ((bi % B_BLOCK_COL_SIZE) / BLOCK_SIZE) * A_BLOCK_COL_SIZE;

        for (int ai = a_offset; ai < (a_offset + A_BLOCK_COL_SIZE);
             ai += B_BLOCK_ROWS) {
            const int c_row = (ai - a_offset) / B_BLOCK_ROWS;

            for (int k = 0; k < B_BLOCK_ROWS; ++k) {
                //printf("%d - C: %d %d, A: %d, B: %d\n", count++, c_row, c_col,
                       //ai + k, bi + k);

                C[(c_col * lda) + c_row] += ARD[ai + k] * BDR[bi + k];
            }
        }
    }
    //puts("------");
    free(ARD);
}
*/

#define SIMD_VEC_SZ 16

#define B_BLOCK_ROWS (2)
#define B_BLOCK_COLS (2)

#define TEST_MAT_LDA 6

double mat[TEST_MAT_LDA][TEST_MAT_LDA];
double mat2[TEST_MAT_LDA][TEST_MAT_LDA];

static void inv_mat(const int lda, const double* __restrict__ S,
                    double* __restrict__ D) {
    for (int col = 0; col < lda; ++col) {
        for (int row = 0; row < lda; ++row) {
            D[(row * lda) + col] = S[(col * lda) + row];
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

static void unroll_rd(const int lda, const double* __restrict__ S,
                      double* __restrict__ D) {
    const int BLOCK_COL_SIZE = lda * B_BLOCK_COLS;

    int c_offset = 0;
    for (int c = 0; c < (lda / B_BLOCK_COLS) * B_BLOCK_COLS; ++c) {
        const int BLOCK_COL_IND = (c / B_BLOCK_COLS) * BLOCK_COL_SIZE;
        for (int r = 0; r < (lda / B_BLOCK_ROWS) * B_BLOCK_ROWS; ++r) {
            int ind = BLOCK_COL_IND + (r * B_BLOCK_COLS) + c_offset;
            D[ind] = S[(c * lda) + r];
        }

        c_offset = (c_offset + 1) * (c_offset < (B_BLOCK_COLS - 1));
    }
}

static void unroll_dr(const int lda, const double* __restrict__ S,
                      double* __restrict__ D) {
    const int BLOCK_SIZE = B_BLOCK_ROWS * B_BLOCK_COLS;
    const int BLOCK_COL_SIZE = lda * B_BLOCK_COLS;

    int r_offset = 0;
    for (int c = 0; c < (lda / B_BLOCK_COLS) * B_BLOCK_COLS; ++c) {
        const int BLOCK_COL_IND = ((c / B_BLOCK_COLS) * BLOCK_COL_SIZE) +
                                  ((c % B_BLOCK_COLS) * B_BLOCK_ROWS);

        for (int r = 0; r < (lda / B_BLOCK_ROWS) * B_BLOCK_ROWS; ++r) {
            int ind =
                BLOCK_COL_IND + ((r / B_BLOCK_ROWS) * BLOCK_SIZE) + r_offset;

            D[ind] = S[(c * lda) + r];
            r_offset = (r_offset + 1) * (r_offset < (B_BLOCK_ROWS - 1));
        }
    }
}

void test_mat_order_ops() {
    /*inv_mat(6, mat, mat2);*/
    for (int i = 0; i < TEST_MAT_LDA; ++i) {
        for (int j = 0; j < TEST_MAT_LDA; ++j)
            mat[i][j] = (i * TEST_MAT_LDA) + j;
    }

    // unroll_rd(TEST_MAT_LDA, mat, mat2);
    // unroll_dr(TEST_MAT_LDA, mat, mat2);
    inv_mat_inplace(TEST_MAT_LDA, mat);

    for (int i = 0; i < TEST_MAT_LDA; ++i) {
        for (int j = 0; j < TEST_MAT_LDA; ++j) printf("%.1f ", mat[i][j]);
        printf("\n");
    }
}

void test_simd() {
    __m256d dot_prod;
    __m256d v1[4], v2[4];

    
    double a[SIMD_VEC_SZ] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    double b[SIMD_VEC_SZ] = {17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                             25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0};

    double control = 0;
    for (int i = 0; i < SIMD_VEC_SZ; ++i) control += a[i] * b[i];
    
    /*
    v1[0] = _mm256d_loadu_pd(a);
    v2[0] = _mm256d_loadu_pd(b);
    v1[0] = _mm256_mul_pd(v1[0], v2[0]);
    */

    printf("v2 %f %f %f %f\n", v2[0], v2[1], v2[2], v2[3]);

    printf("control %f\n", control);
}

int main() { test_simd(); }
