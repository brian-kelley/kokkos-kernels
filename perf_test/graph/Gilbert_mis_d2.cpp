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
#include <Kokkos_Random.hpp>

#include "KokkosKernels_Utils.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_default_types.hpp"

template<typename matrix_type>
Kokkos::View<int*> mis_2(const matrix_type& g)
{
    using sgp_vid_t = typename matrix_type::ordinal_type;
    using sgp_eid_t = typename matrix_type::non_const_size_type;
    using pool_t = Kokkos::Random_XorShift64_Pool<typename matrix_type::execution_space>;
    using gen_t = typename pool_t::generator_type;
    using vtx_view_t = typename matrix_type::index_type::non_const_type;

    sgp_vid_t n = g.numRows();
    Kokkos::View<int*> state("is membership", n);
    sgp_vid_t unassigned_total = n;
    Kokkos::View<uint64_t*> randoms("randomized", n);
    pool_t rand_pool(std::time(nullptr));
    Kokkos::parallel_for("create random entries", n,
        KOKKOS_LAMBDA(sgp_vid_t i){ gen_t generator = rand_pool.get_state(); randoms(i) = generator.urand64();
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
                sgp_vid_t max_idx = tuple_idx(i);
                for (sgp_eid_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
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
    using device_t = Kokkos::DefaultExecutionSpace;
    using size_type = default_size_type;
    using lno_t = default_lno_t;
    using exec_space = typename device_t::execution_space;
    using mem_space = typename device_t::memory_space;
    using crsMat_t = typename KokkosSparse::CrsMatrix<default_scalar, default_lno_t, device_t, void, default_size_type>;
    using lno_view_t = typename crsMat_t::index_type::non_const_type;

    Kokkos::initialize();
    {
      crsMat_t A = KokkosKernels::Impl::read_kokkos_crst_matrix<crsMat_t>(argv[1]);
      //Compute MIS-2
      Kokkos::Timer timer;
      auto mis = mis_2(A);
      double t = timer.seconds();
      auto misHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mis);
      std::vector<int> misList;
      FILE* out = fopen("mis2.txt", "w");
      int misSize = 0;
      for(int i = 0; i < A.numRows(); i++)
      {
        if(misHost(i) == 1)
        {
          fprintf(out, "%d ", i);
          misSize++;
        }
      }
      fclose(out);
      std::cout << "Computed MIS in " << t << " s.\n";
      std::cout << "MIS size: " << misSize << '\n';
    }
    Kokkos::finalize();
    return 0;
}

