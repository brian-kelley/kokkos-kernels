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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KS_MARCHING_HASH_TABLE_HPP
#define KS_MARCHING_HASH_TABLE_HPP

#include "Kokkos_Core.hpp"
#include "Kokkos_ArithTraits.hpp"

namespace KokkosSparse {
namespace Impl {
  struct InsertOrTag {};
  struct InsertAddTag {};
  
  template<typename Key, typename Value>
  struct MarchingHashTable
  {
    using KAT = Kokkos::ArithTraits<Key>;

    //32-bit xorshift hash function
    template<typename T>
    static KOKKOS_INLINE_FUNCTION int hash(T key)
    {
      int k = int(key);
      k ^= k << 13;
      k ^= k >> 17;
      k ^= k << 5;
      return int(k);
    }

    //General insert function.
    //If insertion into empty cell succeeds, returns Key::max.
    //If there are no empty cells, but insertion via eviction succeeds, returns the evicted key.
    //If all cells already have lesser keys, does not insert and returns k.
    template<typename OpTag>
    KOKKOS_INLINE_FUNCTION Key insert(Key k, Value v) {}

    //Specialization that joins value using bitwise-OR.
    template<>
    KOKKOS_INLINE_FUNCTION Key insert<InsertOrTag>(Key k, Value v)
    {
      const Key PENDING = KAT::max() - 1;
      const Key EMPTY = KAT::max();
      int h = hash(k);
      Key maxKey = KAT::zero();
      int maxCell = 0;
      //Need to have a retry mechanism - if multiple threads attempt to evict the same
      //key, only one of them can succeed and it effectively locks that cell while overwriting the value.
      while(true)
      {
        for(int attempt = 0; attempt < nprobe; attempt++)
        {
          int cell = h & (n - 1);
          Key current = keys[cell];
          if(current == EMPTY)
          {
            if(Kokkos::atomic_compare_exchange_strong(&keys[cell], EMPTY, k))
            {
              //Insertion succeeded - update value and done
              Kokkos::atomic_or(&values[cell], v);
              return EMPTY;
            }
            //Otherwise, another thread already grabbed this cell.
            //This is OK - just keep trying other cells.
          }
          else if(current > maxKey)
          {
            maxKey = current;
            maxCell = cell;
          }
          h = hash(h);
        }
        //If here, no empty cells were available. 
      }
    }

    //Specialization that joins value using addition.
    template<>
    KOKKOS_INLINE_FUNCTION Key insert<InsertAddTag>(Key k, Value v)
    {
    }

    Key* keys;
    Value* values;
    int nprobe; //Number of rehashing attempts to use
    int n;      //Length of keys/values (power of 2)
  };
}
}

#endif

