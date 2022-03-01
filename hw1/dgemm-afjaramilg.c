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
        for (int row_a = 0; row_a < B_BLOCK_ROWS; ++row_a) {
            for (int k = 0; k < B_BLOCK_ROWS; ++k) {
                // double b_block_val = B_BUFFER[(col_b * B_BLOCK_ROWS) +
                // row_b];
                double b_block_val = B[(col_b * lda) + k];

                C[(col_b * lda) + row_a] += A[(row_a * lda) + k] * b_block_val;
            }
        }
    }
}

static void do_block(const int lda, const double* A, const double* B,
                     double* __restrict__ C, const int rows_ba) {
    for (int col_b = 0; col_b < B_BUFFER_COLS; ++col_b) {
        for (int row_a = 0; row_a < rows_ba; ++row_a) {
            for (int k = 0; k < B_BUFFER_ROWS; ++k) {
                // double b_block_val = B_BUFFER[(col_b * B_BUFFER_ROWS) +
                // row_b];
                double b_block_val = B[(col_b * lda) + k];

                C[(col_b * lda) + row_a] += A[(row_a * lda) + k] * b_block_val;
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
        B_BUFFER_COLS = min(B_BLOCK_COLS, lda - col_b);

        for (int row_a = 0; row_a < lda; row_a += B_BLOCK_COLS) {
            int rows_ba = min(B_BLOCK_COLS, lda - row_a);

            for (int k = 0; k < lda; k += B_BLOCK_ROWS) {
                B_BUFFER_ROWS = min(B_BLOCK_ROWS, lda - k);

                const double* B_CPY = B + ((col_b * lda) + k);

                // for (int cb = 0; cb < B_BUFFER_COLS; ++cb) {
                // for (int rb = 0; rb < B_BUFFER_ROWS; ++rb) {
                // B_BUFFER[(cb * B_BUFFER_ROWS) + rb] =
                // B_CPY[(cb * lda) + rb];
                //}
                //}
                // printf("b:%i,%i  a:%i,%i\n", row_b, col_b, row_a, row_b);

                if (B_BUFFER_ROWS == B_BLOCK_ROWS &&
                    B_BUFFER_COLS == B_BLOCK_COLS && rows_ba == B_BLOCK_COLS)
                    do_block_fixed(lda, A + ((row_a * lda) + k), B_CPY,
                                   C + ((col_b * lda) + row_a));
                else
                    do_block(lda, A + ((row_a * lda) + k), B_CPY,
                             C + ((col_b * lda) + row_a), rows_ba);
            }
        }
    }

    inv_mat_inplace(lda, A);
    // puts("----------------------------");
}
