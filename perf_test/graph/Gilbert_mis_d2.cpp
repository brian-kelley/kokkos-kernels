#include <stdlib.h>
#include <string>
#include <set>
#include <unistd.h>

#include <iostream>
#include <iomanip>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

#include "KokkosKernels_Utils.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_default_types.hpp"

Kokkos::View<int*> mis_2(const matrix_type& g)
{
    sgp_vid_t n = g.numRows();
    Kokkos::View<int*> state("is membership", n);
    sgp_vid_t unassigned_total = n;
    Kokkos::View<uint64_t*> randoms("randomized", n);
    pool_t rand_pool(std::time(nullptr));
    Kokkos::parallel_for("create random entries", n, KOKKOS_LAMBDA(sgp_vid_t i){
        gen_t generator = rand_pool.get_state();
        randoms(i) = generator.urand64();
        rand_pool.free_state(generator);
    });
    while (unassigned_total > 0)
    {
        Kokkos::View<int*> tuple_state("tuple state", n);
        Kokkos::View<uint64_t*> tuple_rand("tuple rand", n);
        vtx_view_t tuple_idx("tuple index", n);        Kokkos::View<int*> tuple_state_update("tuple state", n);
        Kokkos::View<uint64_t*> tuple_rand_update("tuple rand", n);
        vtx_view_t tuple_idx_update("tuple index", n);
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(const sgp_vid_t i){
            tuple_state(i) = state(i);
            tuple_rand(i) = randoms(i);
            tuple_idx(i) = i;
        });
        for (int k = 0; k < 2; k++) {
                Kokkos::parallel_for(n, KOKKOS_LAMBDA(const sgp_vid_t i){
                int max_state = tuple_state(i);
                uint64_t max_rand = tuple_rand(i);
                sgp_vid_t max_idx = tuple_idx(i);                for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                    sgp_vid_t v = g.graph.entries(j);
                    bool is_max = false;
                    if (tuple_state(v) > max_state) {
                        is_max = true;
                    }
                    else if (tuple_state(v) == max_state) {
                        if (tuple_rand(v) > max_rand) {
                            is_max = true;
                        }
                        else if (tuple_rand(v) == max_rand) {
                            if (tuple_idx(v) > max_idx) {
                                is_max = true;
                            }
                        }
                    }
                    if (is_max) {
                        max_state = tuple_state(v);
                        max_rand = tuple_rand(v);
                        max_idx = tuple_idx(v);
                    }
                }
                tuple_state_update(i) = max_state;
                tuple_rand_update(i) = max_rand;
                tuple_idx_update(i) = max_idx;
            });            Kokkos::parallel_for(n, KOKKOS_LAMBDA(const sgp_vid_t i){
                tuple_state(i) = tuple_state_update(i);
                tuple_rand(i) = tuple_rand_update(i);
                tuple_idx(i) = tuple_idx_update(i);
            });
        }        unassigned_total = 0;
        Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const sgp_vid_t i, sgp_vid_t& thread_sum){
            if (state(i) == 0) {
                if (tuple_state(i) == state(i) && tuple_rand(i) == randoms(i) && tuple_idx(i) == i) {
                    state(i) = 1;
                }
                else if(tuple_state(i) == 1) {
                    state(i) = -1;
                }
            }
            if (state(i) == 0) {
                thread_sum++;
            }
        }, unassigned_total);
    }
    return state;
}

int main(int argc, char *argv[])
{
    MIS2Parameters params;

    if(parse_inputs(params, argc, argv))
    {
        return 1;
    }

    if(params.mtx_file == NULL)
    {
        std::cerr << "Provide a matrix file" << std::endl;
        return 0;
    }

    Kokkos::initialize();

    bool run = false;

    #if defined(KOKKOS_ENABLE_OPENMP)
    if(params.use_openmp)
    {
      run_mis2<Kokkos::OpenMP>(params);
      run = true;
    }
    #endif

    #if defined(KOKKOS_ENABLE_THREADS)
    if(params.use_threads)
    {
      run_mis2<Kokkos::Threads>(params);
      run = true;
    }
    #endif

    #if defined(KOKKOS_ENABLE_CUDA)
    if(params.use_cuda)
    {
      run_mis2<Kokkos::Cuda>(params);
      run = true;
    }
    #endif

    #if defined(KOKKOS_ENABLE_SERIAL)
    if(params.use_serial)
    {
      run_mis2<Kokkos::Serial>(params);
      run = true;
    }
    #endif

    if(!run)
    {
      std::cerr << "*** ERROR: did not run, none of the supported device types were selected.\n";
    }

    Kokkos::finalize();

    return 0;
}
