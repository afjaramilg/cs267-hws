#include "dcomp2d1.h"

dcomp2d1_block::dcomp2d1_block() { omp_init_lock(&lock); }

inline void dcomp2d1_block::push(int ind) {
    omp_set_lock(&lock);
    p.push_back(ind);
    omp_unset_lock(&lock);
}

int dcomp2d1_block::size() const { return p.size(); }

// ==============================================

dcomp2d1::dcomp2d1(int side_p, double cutoff_p, double space_size)
    : side(side_p), cutoff(cutoff_p) {
    grid = new dcomp2d1_block[side * side];
    borders = new dcomp2d1_block[8 * side * side];
    block_lims = new double[side];

    step = space_size / side;
    block_lims[0] = step;

    for (int i = 1; i < side; ++i) {
        block_lims[i] = block_lims[i - 1] + step;
    }
}

dcomp2d1::~dcomp2d1() {
    if (grid != nullptr) delete[] grid;
    if (block_lims != nullptr) delete[] block_lims;
    if (borders != nullptr) delete[] borders;
}

void dcomp2d1::assign(particle_t &p, int ind) {
    int row = find_block(p.y);
    int col = find_block(p.x);
    int block_ind = (row * side) + col;

    // printf("%f %f - %i %i\n", p.y, p.x, row, col);

    grid[block_ind].push(ind);

    dcomp2d1_block *b = borders + (8 * block_ind);
    double bu = block_lims[row];
    double bd = bu - step;
    double br = block_lims[col];
    double bl = br - step;

    uint8_t mask = 0;
    mask |= ((br - p.x) <= cutoff) * 1;  // in right border
    mask |= ((p.y - bd) <= cutoff) * 2;  // in lower border
    mask |= ((p.x - bl) <= cutoff) * 4;  // in left border
    mask |= ((bu - p.y) <= cutoff) * 8;  // in upper border

    switch (mask) {
        case 9:  // right and up
            b[0].push(ind);
            break;
        case 1:  // right
            b[1].push(ind);
            break;
        case 3:  // right and down
            b[2].push(ind);
            break;
        case 2:  // down
            b[3].push(ind);
            break;
        case 6:  // left and down
            b[4].push(ind);
            break;
        case 4:  // left
            b[5].push(ind);
            break;
        case 12:  // left and up
            b[6].push(ind);
            break;
        case 8:  // up
            b[7].push(ind);
            break;
    }
}

int dcomp2d1::find_block(double x) {
    // linear search
    // int l = 0;
    // while (block_lims[l] < x && l < side) ++l;

    // binary search

    int l = 0, r = side;

    while (l < r) {
        int m = (l + r) / 2;
        if (block_lims[m] < x) {
            l = m + 1;
        } else {
            r = m;
        }
    }

    return l;
}
