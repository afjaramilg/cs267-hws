/**
 * https://stackoverflow.com/questions/34165099/in-c-why-is-signed-int-faster-than-unsigned-int
 * https://stackoverflow.com/questions/227897/how-to-allocate-aligned-memory-only-using-the-standard-library
 * https://stackoverflow.com/questions/11714968/how-to-allocate-a-32-byte-aligned-memory-in-c
 * https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#
 */

#include <immintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define B_BLOCK_ROWS (8)
#define B_BLOCK_COLS (8)
#define BLOCK_SIZE (B_BLOCK_ROWS * B_BLOCK_COLS)

const char* dgemm_desc =
    "dgemm using blocked algorithm, but with loops in different order";

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

    /*memset(C, 0, lda * lda * sizeof(double));*/

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
                /*printf("%d - C: %d %d, A: %d, B: %d\n", count++, c_row, c_col,*/
                       /*ai + k, bi + k);*/

                C[(c_col * lda) + c_row] += ARD[ai + k] * BDR[bi + k];
            }
        }
    }
    /*puts("------");*/
    free(ARD);
}
