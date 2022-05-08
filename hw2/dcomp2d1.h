#ifndef DCOMP2D_H
#define DCOMP2D_H

#include <omp.h>
#include <stdio.h>

#include <cstdint>
#include <vector>

#include "common.h"

struct dcomp2d1_block {
    std::vector<int> p;
    omp_lock_t lock;

    dcomp2d1_block();

    void push(int ind);
    int size() const;
};

struct dcomp2d1 {
    int side;
    dcomp2d1_block *grid, *borders;
    double *block_lims;
    double step, cutoff;

    dcomp2d1(int side_p, double cutoff_p, double space_size);
    ~dcomp2d1();

    void assign(particle_t &p, int ind);

   private:
    int find_block(double x);
};

#endif
