#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "dcomp2d1.h"

//
//  benchmarking program
//
int main(int argc, char **argv) {
    int navg, nabsavg = 0;
    double davg, dmin, absmin = 1.0, absavg = 0.0;

    if (find_option(argc, argv, "-h") >= 0) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);

    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t *)malloc(n * sizeof(particle_t));
    double space_size = set_size(n);
    init_particles(n, particles);

    // my parameters----------------------
    const int GRID_SIDE = 200;
    const double CUTOFF = 0.01;
    dcomp2d1 dcomp(GRID_SIDE, CUTOFF, space_size);
    // - ----------------------------------

    // neighbour pointer comparison delta
    const int64_t ptr_cd[8] = {
        -(GRID_SIDE - 1), 1,  GRID_SIDE + 1,    GRID_SIDE,
        GRID_SIDE - 1,    -1, -(GRID_SIDE + 1), -GRID_SIDE};

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();

    for (int step = 0; step < NSTEPS; step++) {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;

        //
        // assign to block
        //
        for (int i = 0; i < n; ++i) {
            dcomp.assign(particles[i], i);
        }
        
        // calculate collisions
       
       
        for (int row = 0; row < GRID_SIDE; ++row) {
            for (int col = 0; col < GRID_SIDE; ++col) {
                dcomp2d1_block *curr = &dcomp.grid[row * GRID_SIDE + col];

                // internal collisions
                for (int i = 0; i < curr->size(); i++) {
                    int piind = curr->p[i];
                    particles[piind].ax = particles[piind].ay = 0;

                    for (int j = 0; j < curr->size(); j++) {
                        int pjind = curr->p[j];
                        apply_force(particles[piind], particles[pjind], &dmin,
                                    &davg, &navg);
                    }

                    
                    for (int k = 0; k < 8; ++k) {
                        dcomp2d1_block *neigh = curr + ptr_cd[k];

                        if (dcomp.grid <= neigh &&
                            neigh < (dcomp.grid + (GRID_SIDE * GRID_SIDE))) {
                            for (int j = 0; j < neigh->size(); ++j) {
                                int pjind = neigh->p[j];
                                apply_force(particles[piind], particles[pjind],
                                            &dmin, &davg, &navg);
                            }
                        }
                    }
                }
            }
        }

        // !! DEBUG !!
        // puts("--------------------");
        // !! DEBUG !!

        //
        //  move particles
        //
        for (int i = 0; i < n; i++) move(particles[i]);
        
        for(int i = 0; i < (GRID_SIDE * GRID_SIDE); ++i) {
            dcomp.grid[i].p.clear();
        }



        if (find_option(argc, argv, "-no") == -1) {
            //
            // Computing statistical data
            //
            if (navg) {
                absavg += davg / navg;
                nabsavg++;
            }
            if (dmin < absmin) absmin = dmin;

            //
            //  save if necessary
            //
            if (fsave && (step % SAVEFREQ) == 0) save(fsave, n, particles);
        }
    }
    simulation_time = read_timer() - simulation_time;

    printf("n = %d, simulation time = %g seconds", n, simulation_time);

    if (find_option(argc, argv, "-no") == -1) {
        if (nabsavg) absavg /= nabsavg;
        //
        //  -the minimum distance absmin between 2 particles during the run of
        //  the simulation -A Correct simulation will have particles stay at
        //  greater than 0.4 (of cutoff) with typical values between .7-.8 -A
        //  simulation were particles don't interact correctly will be less than
        //  0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are
        //  interacting correctly and ~.66 when no particles are interacting
        //
        printf(", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4)
            printf(
                "\nThe minimum distance is below 0.4 meaning that some "
                "particle is not interacting");
        if (absavg < 0.8)
            printf(
                "\nThe average distance is below 0.8 meaning that most "
                "particles are not interacting");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if (fsum) fprintf(fsum, "%d %g\n", n, simulation_time);

    //
    // Clearing space
    //
    if (fsum) fclose(fsum);
    free(particles);
    if (fsave) fclose(fsave);

    return 0;
}
