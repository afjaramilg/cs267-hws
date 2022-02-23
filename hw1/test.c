#include <immintrin.h>
#include <stdio.h>

/*
 * #include <immintrin.h>
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
static void do_block_fixed(const int lda, const double* A, double* __restrict__
C) { for (int col_b = 0; col_b < B_BLOCK_COLS; ++col_b) { for (int row_b = 0;
row_b < B_BLOCK_ROWS; ++row_b) { double b_block_val = B_BUFFER[(col_b *
B_BLOCK_ROWS) + row_b];

            for (int row_a = 0; row_a < B_BLOCK_ROWS; ++row_a) {
                C[(col_b * lda) + row_a] +=
                    A[(row_b * lda) + row_a] * b_block_val;
            }
        }
    }
}

static void do_block(const int lda, const double* A, double* __restrict__ C,
                      const int rows_ba) {
    for (int col_b = 0; col_b < B_BUFFER_COLS; ++col_b) {
        for (int row_b = 0; row_b < B_BUFFER_ROWS; ++row_b) {
            double b_block_val = B_BUFFER[(col_b * B_BUFFER_ROWS) + row_b];

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

            for (int cb = 0; cb < B_BUFFER_COLS; ++cb) {
                for (int rb = 0; rb < B_BUFFER_ROWS; ++rb) {
                    B_BUFFER[(cb * B_BUFFER_ROWS) + rb] =
                        B_CPY[(cb * lda) + rb];
                }
            }

            for (int row_a = 0; row_a < lda; row_a += B_BLOCK_COLS) {
                int rows_ba = min(B_BLOCK_COLS, lda - row_a);

                if (B_BUFFER_ROWS == B_BLOCK_ROWS &&
                    B_BUFFER_COLS == B_BLOCK_COLS && rows_ba == B_BLOCK_COLS)
                    do_block_fixed(lda, A + ((row_b * lda) + row_a),
                             C + ((col_b * lda) + row_a));
                else
                    do_block(lda, A + ((row_b * lda) + row_a),
                              C + ((col_b * lda) + row_a), rows_ba);
            }
        }
    }
}

*/

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

int main() {
    /*inv_mat(6, mat, mat2);*/
    for (int i = 0; i < TEST_MAT_LDA; ++i) {
        for (int j = 0; j < TEST_MAT_LDA; ++j)
            mat[i][j] = (j + 1) * 10 + (i + 1);
    }

    /*unroll_rd(TEST_MAT_LDA, mat, mat2);*/
    unroll_dr(TEST_MAT_LDA, mat, mat2);

    for (int i = 0; i < TEST_MAT_LDA; ++i) {
        for (int j = 0; j < TEST_MAT_LDA; ++j) printf("%.1f ", mat2[i][j]);
        printf("\n");
    }
}
