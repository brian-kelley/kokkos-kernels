#include "Kokkos_Core.hpp"
#include "Kokkos_UnorderedMap.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Device = Kokkos::Device<ExecSpace, typename ExecSpace::memory_space>;

int main()
{
  Kokkos::initialize();
  {
    std::cout << "Testing out UnorderedMap on " << ExecSpace::name() << '\n';
    int n = 10000;
    Kokkos::UnorderedMap<uint64_t, void, Device> myMap(n);
    int numToInsert = n * 0.75;
    //Insert each value in [0..numToInsert-1] 5 times
    int uniqueInserted;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpace>(0, numToInsert * 5), 
    KOKKOS_LAMBDA(int i, int& luniqueInserted)
    {   
      int valueToInsert = i % numToInsert;
      auto result = myMap.insert(valueToInsert);
      if(result.success() && !result.existing())
      {   
        luniqueInserted++;
      }   
      else if(result.failed())
      {   
        printf("Inserting failed!\n");
      }   
    }, uniqueInserted);
    std::cout << "Inserted " << uniqueInserted << " unique values successfully (should be " << numToInsert << ")\n";
  }
  Kokkos::finalize();
  return 0;
}
