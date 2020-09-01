/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Brian Kelley (bmkelle@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

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
#include "KokkosSparse_spadd.hpp"
#include "KokkosGraph_MIS2.hpp"
#include "KokkosKernels_default_types.hpp"

using namespace KokkosGraph;

template<typename lno_t, typename size_type, typename rowmap_t, typename entries_t, typename mis_t>
bool verifyD2MIS(
    lno_t numVerts,
    const rowmap_t& rowmap, const entries_t& entries,
    const mis_t& misArray)
{
  //set a std::set of the mis, for fast membership test
  std::set<lno_t> mis;
  for(size_t i = 0; i < misArray.extent(0); i++)
    mis.insert(misArray(i));
  for(lno_t i = 0; i < numVerts; i++)
  {
    //determine whether another vertex in the set is
    //within 2 hops of i.
    bool misIn2Hops = false;
    for(size_type j = rowmap(i); j < rowmap(i + 1); j++)
    {
      lno_t nei1 = entries(j);
      if(nei1 == i || nei1 >= numVerts)
        continue;
      if(mis.find(nei1) != mis.end())
      {
        misIn2Hops = true;
        break;
      }
      for(size_type k = rowmap(nei1); k < rowmap(nei1 + 1); k++)
      {
        lno_t nei2 = entries(k);
        if(nei2 == i || nei2 >= numVerts)
          continue;
        if(mis.find(nei2) != mis.end())
        {
          misIn2Hops = true;
          break;
        }
      }
    }
    if(mis.find(i) == mis.end())
    {
      //i is not in the set
      if(!misIn2Hops)
      {
        std::cout << "INVALID D2 MIS: vertex " << i << " is not in the set,\n";
        std::cout << "but there are no vertices in the set within 2 hops.\n";
        return false;
      }
    }
    else
    {
      //i is in the set
      if(misIn2Hops)
      {
        std::cout << "INVALID D2 MIS: vertex " << i << " is in the set,\n";
        std::cout << "but there is another vertex within 2 hops which is also in the set.\n";
        return false;
      }
    }
  }
  return true;
}

void check_mis2(const char* matFile, const char* misFile)
{
  using device_t = Kokkos::DefaultHostExecutionSpace;
  using size_type = default_size_type;
  using lno_t = default_lno_t;
  using exec_space = typename device_t::execution_space;
  using mem_space = typename device_t::memory_space;
  using crsMat_t = typename KokkosSparse::CrsMatrix<default_scalar, default_lno_t, device_t, void, default_size_type>;
  using lno_view_t = typename crsMat_t::index_type::non_const_type;
  using KKH = KokkosKernels::Experimental::KokkosKernelsHandle<size_type, lno_t, double, exec_space, mem_space, mem_space>;

  Kokkos::Timer t;
  crsMat_t A = KokkosKernels::Impl::read_kokkos_crst_matrix<crsMat_t>(matFile);
  std::cout << "I/O time: " << t.seconds() << " s\n";
  t.reset();
  auto rowmap = A.graph.row_map;
  auto entries = A.graph.entries;
  lno_t numVerts = A.numRows();

  std::cout << "Num verts: " << numVerts << '\n'
            << "Num edges: " << A.nnz() << '\n';

  std::vector<lno_t> misList;

  {
    int tmp;
    FILE* mis = fopen(misFile, "r");
    while(1 == fscanf(mis, "%d", &tmp))
    {
      misList.push_back(tmp);
    }
    fclose(mis);
  }

  lno_view_t mis("asdf", misList.size());
  memcpy(mis.data(), misList.data(), misList.size() * sizeof(lno_t));

  if(verifyD2MIS
    <lno_t, size_type, decltype(rowmap), decltype(entries), decltype(mis)>
    (numVerts, rowmap, entries, mis))
  {
    std::cout << "MIS-2 is correct.\n";
  }
  else
    std::cout << "*** MIS-2 not correct! ***\n";
}

int main(int argc, char *argv[])
{
  Kokkos::initialize();
  check_mis2(argv[1], argv[2]);
  Kokkos::finalize();
  return 0;
}

