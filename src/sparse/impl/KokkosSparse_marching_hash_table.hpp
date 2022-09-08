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
#include <numeric>

namespace KokkosSparse {
namespace Impl {
  struct InsertOrTag {};
  struct InsertAddTag {};
  
  template<typename Key, typename Value>
  struct MarchingHashTable
  {
    using KAT = Kokkos::ArithTraits<Key>;
    using VAT = Kokkos::ArithTraits<Value>;

    MarchingHashTable(Key* k, Value* v, int n_, int nprobe_)
      : keys(k), values(v), n(n_), nprobe(nprobe_)
    {}

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
    //It just attempts to insert key k. Keys may be evicted at any time so this function
    //can't determine whether the key will stick. In the value update functions, the keys
    //are all held constant so they will check instead.
    //
    //If this evicts a key, the evicted key is returned. Otherwise ~0 (-1 for signed key).
    KOKKOS_INLINE_FUNCTION Key insert(Key k)
    {
      const Key EMPTY = KAT::max();
      //Need to have a retry mechanism - if multiple threads attempt to evict the same
      //key, only one of them can succeed.
      while(true)
      {
        int h = hash(k);
        Key maxKey = KAT::zero();
        int maxCell = 0;
        for(int attempt = 0; attempt < nprobe; attempt++)
        {
          int cell = h & (n - 1);
          //Another retry loop - needed if keys[cell] changes after reading it
          while(true)
          {
            Key current = Kokkos::atomic_load(&keys[cell]);
            if(current == k)
            {
              //Key already present.
              //Note that it may still be evicted at any time, but that is OK
              return ~Key(0);
            }
            else if(current == EMPTY)
            {
              //This cell is empty, so if k were already present, it would have been seen by now
              if(Kokkos::atomic_compare_exchange_strong(&keys[cell], EMPTY, k))
              {
                //Insertion succeeded - initialize value and done
                values[cell] = VAT::zero();
                return ~Key(0);
              }
              //If cmp-exch fails, another thread already put a key in this cell.
              //This is OK - just keep trying until keys[cell] settles
              //on either k or a different key.
            }
            else if(current > maxKey)
            {
              //Cell is occupied. Keep track of maximum key over all of k's possible cells.
              maxKey = current;
              maxCell = cell;
              break;
            }
          }
          h = hash(h);
        }
        //If here, no empty cells were available. Try evicting max key, but only if k is an improvement (is less than that max)
        if(k < maxKey)
        {
          //But don't just write k there, only place it there if maxKey has not already been evicted by other thread
          if(Kokkos::atomic_compare_exchange_strong(&keys[maxCell], maxKey, k))
          {
            values[maxCell] = VAT::zero();
            return maxKey;
          }
          //If this cmp-exch fails, have to go through the cells again to recompute maxKey
        }
        else
        {
          //Max key is less than k, so give up on inserting k.
          return ~Key(0);
        }
      }
    }

    //Look for key k, and join v into the corresponding value using bitwise OR.
    //Note that all keys are now held constant, and only values are changing.
    //Returns true if key found and value updated, false if key not found.
    //This is only supported when Value is an integer type.
    KOKKOS_INLINE_FUNCTION bool updateValueOr(Key k, Value v,
        typename std::enable_if<std::numeric_limits<Value>::is_integer>::type* = nullptr)
    {
      int h = hash(k);
      for(int attempt = 0; attempt < nprobe; attempt++)
      {
        int cell = h & (n - 1);
        if(keys[cell] == k)
        {
          Kokkos::atomic_fetch_or(&values[cell], v);
          return true;
        }
        h = hash(h);
      }
      return false;
    }

    //Same as updateValueOr, but instead updates values using addition.
    KOKKOS_INLINE_FUNCTION bool updateValueAdd(Key k, Value v)
    {
      int h = hash(k);
      for(int attempt = 0; attempt < nprobe; attempt++)
      {
        int cell = h & (n - 1);
        if(keys[cell] == k)
        {
          Kokkos::atomic_fetch_add(&values[cell], v);
          return true;
        }
        h = hash(h);
      }
      return false;
    }

    Key* keys;
    Value* values;
    int n;      //Length of keys/values (power of 2)
    int nprobe; //Number of rehashing attempts to use
  };
}
}

#endif

