#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "common.h"
#define cutoff 0.01

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
    set_size(n);
    init_particles(n, particles);

    auto sort_by_x = [](particle_t &a, particle_t &b) {
        return (a.x == b.x) ? a.y < b.y : a.x < b.x;
    };

    std::vector<std::vector<int>> slices;

    long long useless_comps = 0;

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();

    for (int step = 0; step < NSTEPS; step++) {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;

        std::sort(particles, particles + n, sort_by_x);

        double last_x = -1.0;
        for (int i = 0; i < n; ++i) {
            particle_t &pi = particles[i];
            pi.ax = pi.ay = 0;

            if (pi.x > last_x) {
                slices.push_back({});
                last_x = pi.x;
            }

            slices.back().push_back(i);
        }


        for (int i = 1; i < slices.size(); ++i) {
            //printf("slice %i size %i\n", i, slices[i].size());

            int pa_ind = slices[i - 1][0];
            int pb_ind = slices[i][0];

            particle_t &pa = particles[pa_ind];
            particle_t &pb = particles[pb_ind];

            if (pb.x <= pa.x) puts("HELLLOOOO");

            for (int j = 1; j < slices[i].size(); ++j) {
                pa_ind = slices[i][j - 1];
                pb_ind = slices[i][j];

                pa = particles[pa_ind];
                pb = particles[pb_ind];

                if (pb.y < pa.y) printf("%f <= %f\n", pb.y, pa.y);
            }
        }

        for (int ia = 0; ia < slices.size(); ++ia)
            for (int ja = 0; ja < slices[ia].size(); ++ja) {
                int pa_ind = slices[ia][ja];
                particle_t &pa = particles[pa_ind];

                for (int ib = ia; ib < slices.size(); ++ib) {
                    for (int jb = ja; jb < slices[ib].size(); ++jb) {
                        int pb_ind = slices[ib][jb];
                        particle_t &pb = particles[pb_ind];

                        if ((pb.y - pa.y) > cutoff) break;

                        useless_comps +=
                            apply_force(pa, pb, &dmin, &davg, &navg);
                        useless_comps +=
                            apply_force(pb, pa, &dmin, &davg, &navg);
                    }

                    int pib0_ind = slices[ib][0];
                    particle_t &pib0 = particles[pib0_ind];

                    if ((pib0.x - pa.x) > cutoff) break;
                }
            }

        slices.clear();

        /*
        for (int i = 0; i < n; ++i) {
            particle_t &pi = particles[i];

            for (int j = i; j < n; ++j) {
                particle_t &pj = particles[j];
                if ((pj.x - pi.x) <= cutoff) {
                    apply_force(pi, pj, &dmin, &davg, &navg);
                    apply_force(pj, pi, &dmin, &davg, &navg);
                }
            }
        }
        */

        //
        //  move particles
        //
        for (int i = 0; i < n; i++) move(particles[i]);

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
    printf("useless comparison %lld\n", useless_comps);

    //
    // Clearing space
    //
    if (fsum) fclose(fsum);
    free(particles);
    if (fsave) fclose(fsave);

    return 0;
}
