#include "KokkosGraph_wiki_9pt_stencil.hpp"
#include "KokkosGraph_Matching.hpp"

int main(int argc, char* argv[])
{
  Kokkos::initialize();
  bool unicode = false;
  if(argc == 2 && !strcmp(argv[1], "--unicode"))
    unicode = true;
  {
    using GraphDemo::numVertices;
    RowmapType rowmapDevice;
    ColindsType colindsDevice;
    //Step 1: Generate the graph on host, allocate space on device, and copy.
    //See function "generate9pt" below.
    GraphDemo::generate9pt(rowmapDevice, colindsDevice);
    //Step 2: Run maximal matching and display the result
    {
      Ordinal numClusters = 0;
      auto matches = KokkosGraph::Experimental::graph_match<ExecSpace, RowmapType, ColindsType>(
          rowmapDevice, colindsDevice);
      //Count the matches.
      auto matchesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matches);
      int numMatches = 0;
      for(size_t i = 0; i < matchesHost.extent(0); i++)
      {
        if(i != (size_t) matchesHost(i))
          numMatches++;
      }
      std::cout << "Matched " << numMatches * 2 << " vertices out of " << numVertices << '\n';
      //coarsening labels can be printed in the same way as colors
      std::cout << "Grid with connections between matched vertices:\n\n";
      GraphDemo::printMatching(matches, unicode);
      putchar('\n');
      if(!unicode)
        std::cout << "NOTE: if your terminal can display Unicode, use --unicode flag.\n";
    }
  }
  Kokkos::finalize();
  return 0;
}

