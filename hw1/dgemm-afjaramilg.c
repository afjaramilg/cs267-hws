/**
 * https://stackoverflow.com/questions/34165099/in-c-why-is-signed-int-faster-than-unsigned-int
 * https://stackoverflow.com/questions/227897/how-to-allocate-aligned-memory-only-using-the-standard-library
 * https://stackoverflow.com/questions/11714968/how-to-allocate-a-32-byte-aligned-memory-in-c
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#
 */

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