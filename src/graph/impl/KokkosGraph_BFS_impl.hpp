/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOSGRAPH_BFS_IMPL_H
#define KOKKOSGRAPH_BFS_IMPL_H

#include "KokkosKernels_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Sort.hpp>

//The BFS functor, with explicit level sets
//Level set format is like CRS:
//  if v = levelVerts(j) is in level L, levelOffsets(L) <= j < levelOffsets(L+1)
//The root is always the only vertex in level 0.
template<typename Ordinal, typename Offset, typename Rowmap, typename Colinds, typename OrdinalView, typename Worklist>
struct BfsLevelSetFunctor
{
  KOKKOS_INLINE_FUNCTION void operator()(const Ordinal i, Ordinal& lcount, bool finalPass) const
  {
    Ordinal v = worklist(i, w);
    Offset rowBegin = rowmap(v);
    Offset rowEnd = rowmap(v + 1);
    for(Offset j = rowBegin; j < rowEnd; j++)
    {
      Ordinal nei = colinds(j);
      if(v == nei)
        continue;
      bool isPred = false;
      if(pred(nei) == v)
      {
        lcount++;
        isPred = true;
      }
      else if(pred(nei) == nv)
      {
        //This neighbor has not been assigned a predecessor yet,
        //so attempt to use this vertex
        if(nv == Kokkos::atomic_compare_exchange(&pred(nei), nv, v))
        {
          //cmp-xchg uniquely succeeded for this vertex, so v is predecssor to nei
          lcount++;
          isPred = true;
        }
      }
      if(finalPass && isPred)
      {
        //v is the unique predecessor of nei, and v is in the current frontier.
        //Therefore nei is in the next frontier, and lcount gives the index to insert it.
        worklist(lcount, 1 - w) = nei;
        levelVerts(levelSetOffset + lcount) = nei;
      }
    }
    if(finalPass && i == worklistLen - 1)
    {
      //Once per kernel launch, set the next entry of levelOffsets to mark
      //the end of this level set
      levelOffsets(level + 1) = levelOffsets(level) + worklistLen;
    }
  }

  Ordinal nv;
  Ordinal level;
  Ordinal levelSetOffset; //This value is updated on host after each level set is constructed.
  OrdinalView levelOffsets;
  OrdinalView levelVerts;
  Rowmap rowmap;
  Colinds colinds;
  OrdinalView pred;  //Initially all nv, pred(i) marks the predecessor used to bring vert i into a frontier.
  Ordinal worklistLen;
  Worklist worklist; //an nx2 array
  int w;             //selects the current column of worklist
};

//BFS, but instead of building level sets, assign roots to each vertex.
//The first worklist is the list of roots.
//"distances" should be initialized to 0 for roots, and nv for all others.
//  After finishing, distances(v) will hold the unweighted distance to nearest root.
//"vertClusters" should be initialized to rootID for roots, and 'nv' for all others.
//  After finishing, vertClusters(v) is the cluster containing v.
template<typename Ordinal, typename Offset, typename Rowmap, typename Colinds, typename OrdinalView, typename Worklist>
struct ClusterBfsFunctor
{
  KOKKOS_INLINE_FUNCTION void operator()(const Ordinal i, Ordinal& lcount, bool finalPass) const
  {
    Ordinal v = worklist(i, w);
    Offset rowBegin = rowmap(v);
    Offset rowEnd = rowmap(v + 1);
    for(Offset j = rowBegin; j < rowEnd; j++)
    {
      Ordinal nei = colinds(j);
      if(v == nei)
        continue;
      bool isPred = false;
      if(pred(nei) == v)
      {
        lcount++;
        isPred = true;
      }
      else if(pred(nei) == nv)
      {
        //This neighbor has not been assigned a predecessor yet,
        //so attempt to use this vertex
        if(nv == Kokkos::atomic_compare_exchange(&pred(nei), nv, v))
        {
          //cmp-xchg uniquely succeeded for this vertex, so v is predecssor to nei
          lcount++;
          isPred = true;
        }
      }
      if(finalPass && isPred)
      {
        //v is the unique predecessor of nei, and v is in the current frontier.
        //Therefore nei is in the next frontier, and lcount gives the index to insert it.
        worklist(lcount, 1 - w) = nei;
        vertClusters(nei) = vertClusters(v);
        distances(nei) = level;
      }
    }
  }

  Ordinal nv;
  Ordinal level;
  OrdinalView distances;
  OrdinalView vertClusters;
  Rowmap rowmap;
  Colinds colinds;
  OrdinalView pred;  //Initially all nv, pred(i) marks the predecessor used to bring vert i into a frontier.
  Worklist worklist; //an nx2 array
  int w;             //selects the current column of worklist
  OrdinalView distances;  //Mininum distance to root
};

//BFS, but for the SSSP problem (G is directed, weighted, but has no negative cycles).
//This may visit an individual vertex (and update its predecessor) multiple times.
template<typename Ordinal, typename Offset, typename Rowmap, typename Colinds, typename OrdinalView, typename WeightView, typename Worklist>
struct SSSPFunctor
{
  KOKKOS_INLINE_FUNCTION void operator()(const Ordinal i, Ordinal& lcount, bool finalPass) const
  {
    Ordinal v = worklist(i, w);
    Offset rowBegin = rowmap(v);
    Offset rowEnd = rowmap(v + 1);
    for(Offset j = rowBegin; j < rowEnd; j++)
    {
      Ordinal nei = colinds(j);
      if(v == nei)
        continue;
      bool isPred = false;
      if(pred(nei) == v)
      {
        lcount++;
        isPred = true;
      }
      else if(pred(nei) == nv)
      {
        //This neighbor has not been assigned a predecessor yet,
        //so attempt to use this vertex
        if(nv == Kokkos::atomic_compare_exchange(&pred(nei), nv, v))
        {
          //cmp-xchg uniquely succeeded for this vertex, so v is predecssor to nei
          lcount++;
          isPred = true;
        }
      }
      if(finalPass && isPred)
      {
        //v is the unique predecessor of nei, and v is in the current frontier.
        //Therefore nei is in the next frontier, and lcount gives the index to insert it.
        worklist(lcount, 1 - w) = nei;
        vertClusters(nei) = vertClusters(v);
        distances(nei) = level;
      }
    }
  }

  Ordinal nv;
  Ordinal level;
  Rowmap rowmap;
  Colinds colinds;
  WeightView weights;
  OrdinalView pred;  //Initially all nv, pred(i) marks the predecessor used to bring vert i into a frontier.
  WeightView distances;
  Worklist worklist; //an nx2 array
  int w;             //selects the current column of worklist
  OrdinalView distances;  //Mininum distance to root
};

#endif

