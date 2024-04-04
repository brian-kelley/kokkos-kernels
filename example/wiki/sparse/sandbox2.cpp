#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosSparse_Utils.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include "KokkosBlas.hpp"
#include <iostream>
#include <map>
#include <string>

using Device = Kokkos::DefaultHostExecutionSpace;
using Matrix = Kokkos::View<double**, Kokkos::LayoutRight, Device>;
using Vector = Kokkos::View<double*, Kokkos::LayoutLeft, Device>;
using IntVector = Kokkos::View<int*, Kokkos::LayoutLeft, Device>;
using KAT = Kokkos::ArithTraits<double>;

template<typename T>
void print1D(const T& x)
{
  KokkosKernels::Impl::print_1Dview(x);
  std::cout << '\n';
}

template<typename T>
void print2D(const T& A)
{
  int n = A.extent(0);
  for(int i = 0; i < n; i++)
    KokkosKernels::Impl::print_1Dview(Kokkos::subview(A, i, Kokkos::ALL()));
  std::cout << '\n';
}

template<typename T>
void fillRand(const T& v)
{
  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(12345);
  Kokkos::fill_random(v, rand_pool, 0.3, 1.0);
}

Vector randVec(int n)
{
  Vector v("v", n);
  fillRand(v);
  return v;
}

Matrix randMat(int n)
{
  Matrix m("m", n, n);
  fillRand(m);
  return m;
}

Matrix identity(int n)
{
  Matrix I("I", n, n);
  for(int i = 0; i < n; i++)
    I(i, i) = 1;
  return I;
}

// Ax - b
template<typename AT>
Vector residual(const AT& A, const Vector& x, const Vector& b)
{
  auto n = b.extent(0);
  Vector res("res", n);
  Kokkos::deep_copy(res, b);
  KokkosBlas::gemv("N", 1.0, A, x, -1.0, res);
  return res;
}

//Scale v to be unit length.
Vector normalize(const Vector& v)
{
  auto norm = KokkosBlas::nrm2(v);
  Vector vnorm("vnorm", v.extent(0));
  if(KAT::abs(norm) < 1e-15)
    return vnorm;
  KokkosBlas::scal(vnorm, 1.0 / norm, v);
  return vnorm;
}

//Scale v to be unit length.
void normalizeInPlace(const Vector& v)
{
  auto norm = KokkosBlas::nrm2(v);
  KokkosBlas::scal(v, 1.0 / norm, v);
}

struct BMK
{
  BMK(Matrix A_, Vector b_)
    : A(A_), b(b_)
  {
    // n = rank of original system
    n = A.extent(0);
    int numVars = 0;
    int numRows = 0;
    int erowCounter = 0;
    int srowCounter = 0;
    int tsrowCounter = 0;
    // Build Ah as COO initially
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    std::vector<double> bvec;
    using Entry = std::pair<int, int>;
    using EntryVal = std::pair<int, double>;
    std::map<Entry, int> entryToVar;
    // First, create an unknown variable for each entry in A
    for(int c = 0; c < n; c++)
    {
      for(int r = 0; r < n; r++)
      {
        int var = numVars++;
        std::ostringstream oss;
        oss << "X_" << c << "_" << r;
        varLabels.push_back(oss.str());
        entryToVar[Entry(r, c)] = var;
      }
    }
    // Add equality constraints for each column of Ah (each variable in A)
    for(int c = 0; c < n; c++)
    {
      std::vector<int> varsToMakeEqual;
      for(int r = 0; r < n; r++)
      {
        // Get the free variable for which this entry is the coefficient
        varsToMakeEqual.push_back(entryToVar[Entry(r, c)]);
      }
      while(varsToMakeEqual.size() >= 2)
      {
        std::vector<int> nextVarsToMakeEqual;
        for(size_t i = 0; i < varsToMakeEqual.size(); i += 2)
        {
          int hrow = numRows++;
          rowLabels.push_back(std::string("Equal") + std::to_string(erowCounter++));
          int v1 = varsToMakeEqual[i];
          int v2 = varsToMakeEqual[i + 1];
          nextVarsToMakeEqual.push_back(v1);
          rows.push_back(hrow);
          cols.push_back(v1);
          vals.push_back(1);
          rows.push_back(hrow);
          cols.push_back(v2);
          vals.push_back(-1);
          bvec.push_back(0);
        }
        varsToMakeEqual = nextVarsToMakeEqual;
      }
    }
    // Next, for each row of the original matrix, express that
    // <A_row, x> = b_row using a binary tree reduction
    int sumcounter = 0;
    for(int r = 0; r < n; r++)
    {
      double bval = b(r);
      std::vector<EntryVal> valuesToReduce;
      for(int c = 0; c < n; c++)
      {
        // Get the free variable for which this entry is the coefficient
        int hvar = entryToVar[Entry(r, c)];
        double v = A(r, c);
        valuesToReduce.emplace_back(hvar, v);
      }
      // Now build the binary tree sum-reduction by introducing a new variable for each summed pair.
      // Except, when the final pair is summed, it's the original b value
      std::vector<EntryVal> nextValuesToReduce;
      while(true)
      {
        nextValuesToReduce.clear();
        if(valuesToReduce.size() == 2)
        {
          // The last two variables have weighted sum equal to original b
          int hrow = numRows++;
          rowLabels.push_back(std::string("Top_Sum") + std::to_string(tsrowCounter++));
          rows.push_back(hrow);
          cols.push_back(valuesToReduce[0].first);
          vals.push_back(valuesToReduce[0].second);
          rows.push_back(hrow);
          cols.push_back(valuesToReduce[1].first);
          vals.push_back(valuesToReduce[1].second);
          bvec.push_back(bval);
          break;
        }
        // If an odd number of entries, move the last one in front for balancing
        if(valuesToReduce.size() % 2)
        {
          nextValuesToReduce.push_back(valuesToReduce.back());
          valuesToReduce.pop_back();
        }
        // now valuesToReduce is an even number at least 2
        for(size_t i = 0; i < valuesToReduce.size(); i += 2)
        {
          int var1 = valuesToReduce[i].first;
          int var2 = valuesToReduce[i + 1].first;
          double coeff1 = valuesToReduce[i].second;
          double coeff2 = valuesToReduce[i + 1].second;
          int hrow = numRows++;
          rowLabels.push_back(std::string("Sum") + std::to_string(srowCounter++));
          int sumvar = numVars++;
          std::ostringstream oss;
          oss << "t_" << sumcounter++;
          varLabels.push_back(oss.str());
          // coeff1 * var1 + coeff2 * var2 - sumvar = 0
          rows.push_back(hrow);
          cols.push_back(var1);
          vals.push_back(coeff1);
          rows.push_back(hrow);
          cols.push_back(var2);
          vals.push_back(coeff2);
          rows.push_back(hrow);
          cols.push_back(sumvar);
          vals.push_back(-1);
          nextValuesToReduce.emplace_back(sumvar, 1);
          bvec.push_back(0);
        }
        valuesToReduce = nextValuesToReduce;
      }
    }
    std::cout << "Finished assembling hypersparse problem.\n";
    std::cout << "Orig problem has " << n << " unknowns.\n";
    std::cout << "Hypersparse problem has " << numVars << " unknowns and " << rows.size() << " nonzeros.\n";
    std::cout << "Number of rows should match unknowns: " << numRows << '\n';
    nh = numVars;
    bh = Vector("bh", nh);
    for(size_t i = 0; i < bvec.size(); i++)
      bh(i) = bvec[i];
    Ah = Matrix("Ah", nh, nh);
    for(size_t i = 0; i < rows.size(); i++)
      Ah(rows[i], cols[i]) = vals[i];
    //initially, pivot is entry 0,0 (nothing eliminated yet)
    pivot = 0;
  }

  // Display Ah as an undirected graph to show connectivity between variables
  void displayAsGraph(std::string label, int id)
  {
    std::string dotname = std::string("bmk") + std::to_string(id) + ".dot";
    std::string imgname = std::string("bmk") + std::to_string(id) + ".png";
    std::ofstream f(dotname);
    f << "graph {\n";
    f << "label=\"" << label << "\";\n";
    f << "overlap=false;\n";
    // For each nonzero in Ah, produce an edge
    for(int i = pivot; i < nh; i++)
    {
      // Define a "row" node with blue color
      f << rowLabels[i] << " [color=blue];\n";
    }
    for(int i = pivot; i < nh; i++)
    {
      for(int j = pivot; j < nh; j++)
      {
        if(Ah(i,j) != 0.0)
        {
          f << varLabels[j] << " -- " << rowLabels[i] << '\n';
        }
      }
    }
    f << "}\n";
    f.close();
    std::string layoutEngine = "neato";
    std::ostringstream cmd1, cmd2;
    cmd1 << "dot -Tpng -K" << layoutEngine << " " << dotname << " -o " << imgname;
    system(cmd1.str().c_str());
    cmd2 << "firefox " << imgname;
    system(cmd2.str().c_str());
  }

  // Use given row to get an expression for the given variable,
  // and subsitute that variable in all its other uses.
  // This swaps the row to the current pivot position, and
  // then advances pivot by 1.
  void elim(std::string varName, std::string rowName)
  {
    int elimVar = -1;
    for(size_t i = 0; i < varLabels.size(); i++)
    {
      if(varLabels[i] == varName)
      {
        elimVar = i;
        break;
      }
    }
    if(elimVar == -1)
      throw std::invalid_argument("Not a valid var name");
    int pivRow = -1;
    for(size_t i = 0; i < rowLabels.size(); i++)
    {
      if(rowLabels[i] == rowName)
      {
        pivRow = i;
        break;
      }
    }
    if(pivRow == -1)
      throw std::invalid_argument("Not a valid row name");
    if(pivRow < pivot)
      throw std::invalid_argument("Pivot has already moved past that row");
    // Make sure variable to eliminate actually appears in that row
    if(Ah(pivRow, elimVar) == 0.0)
    {
      throw std::runtime_error("Variable to eliminate is zero in that row, cannot use as pivot!");
    }
    // Swap 3 things to swap pivRow to pivot:
    // - row labels
    // - values of Ah
    // - values of bh
    std::swap(rowLabels[pivot], rowLabels[pivRow]);
    for(int i = 0; i < nh; i++)
    {
      std::swap(Ah(pivot, i), Ah(pivRow, i));
    }
    std::swap(bh(pivot), bh(pivRow));
    pivRow = pivot;
    // Do the same to swap column elimVar to pivot
    std::swap(varLabels[pivot], varLabels[elimVar]);
    for(int j = 0; j < nh; j++)
    {
      std::swap(Ah(j, pivot), Ah(j, elimVar));
    }
    // Go down the column under pivot and eliminate all the nonzeros
    double pivval = Ah(pivot, pivot);
    for(int r = pivot + 1; r < nh; r++)
    {
      if(Ah(r, pivot) != 0.0)
      {
        double mult = Ah(r, pivot) / pivval;
        for(int c = pivot; c < nh; c++)
        {
          Ah(r, c) -= mult * Ah(pivot, c);
        }
        bh(r) -= mult * bh(pivot);
      }
    }
    // Now advance pivot to effectively remove 1 row and 1 column from flow graph to display next
    std::cout << "Elimated " << varName << " using row " << rowName << '\n';
    pivot++;
  }

  void elimStepInteractive()
  {
    std::string varName, rowName;
    std::cout << "Enter name of variable to eliminate: ";
    std::cin >> varName;
    std::cout << "Enter name of pivot row: ";
    std::cin >> rowName;
    elim(varName, rowName);
  }

  void interactiveEliminate()
  {
    displayAsGraph("Initial system", 0);
    for(int step = 0;; step++)
    {
      bool success = true;
      do
      {
        try
        {
          elimStepInteractive();
        }
        catch(std::exception& e)
        {
          std::cout << e.what() << '\n';
          std::cout << "Try again\n";
          success = false;
        }
      } while(!success);
      displayAsGraph("Elim step" + std::to_string(step), step+1);
    }
  }

  void elimMultiple(std::vector<std::string> varsToElim, std::vector<std::string> rowsToUse)
  {
    for(size_t i = 0; i < varsToElim.size(); i++)
      elim(varsToElim[i], rowsToUse[i]);
  }

  void elimInterleaved(std::vector<std::string> names)
  {
    for(size_t i = 0; i < names.size(); i += 2)
      elim(names[i], names[i + 1]);
  }

  void noninteractiveEliminate()
  {
    displayAsGraph("Initial system", 0);
    elimInterleaved({"t_1", "Sum1", "X_2_1", "Equal6", "X_3_1", "Sum3", "X_3_0", "Equal9"});
    elimInterleaved({"t_2", "Sum2", "X_0_1", "Equal0", "X_1_0", "Sum0", "X_1_1", "Equal3"});
    elimInterleaved({"t_5", "Sum5", "X_3_2", "Equal10", "X_3_3", "Sum7", "X_2_3", "Equal7"});
    elimInterleaved({"t_6", "Sum6", "X_1_3", "Equal4", "X_1_2", "Sum4", "X_0_3", "Equal1"});
    displayAsGraph("Elim step 1", 1);
    elimInterleaved({"X_2_0", "Equal8", "X_0_2", "Equal2"});
    displayAsGraph("Elim step 2", 2);
    //elimInterleaved({"t_4", "Top_Sum3", "X_0_0", "Equal5", "t_0", "Top_Sum1", "t_3", "Top_Sum0", "X_2_2", "Equal11", "t_7", "Top_Sum2"});
    elimInterleaved({"t_4", "Top_Sum3", "t_0", "Top_Sum1", "X_2_2", "Equal11"});
    displayAsGraph("Elim step 3", 3);
    /*
    elimMultiple(
        {"t_2", "X_3_1", "X_2_1", "X_2_3", "X_1_3", "t_0", "t_4", "t_6", "X_3_3", "X_0_1", "X_1_1", "X_0_3"},
        {"Top_Sum1", "Equal9", "Equal6", "Equal7", "Equal4", "Top_Sum0", "Top_Sum2", "Top_Sum3", "Equal10", "Equal0", "Equal3", "Equal1"});
    displayAsGraph("Elim step 1", 1);
    // Contract upper-left to just X_0_0
    elimMultiple(
        {"X_0_1", "X_1_1", "X_1_0"},
        {"Equal0", "Sum2", "Equal3"});
    // Contract upper-right to just X_2_0
    elimMultiple(
        {"X_2_1", "X_3_1", "X_3_0"},
        {"Equal6", "Sum3", "Equal9"});
    // Contract lower-left to just X_0_2
    elimMultiple(
        {"X_0_3", "X_1_3", "X_1_2"},
        {"Equal1", "Sum6", "Equal4"});
    // Contract lower-right to just X_2_2
    elimMultiple(
        {"X_2_3", "X_3_2", "X_3_3"},
        {"Equal7", "Sum5", "Equal10"});
    displayAsGraph("Elim step 1", 1);
    // Contract intermediate sums
    elimMultiple(
        {"t_2", "t_3", "t_6", "t_7"},
        {"Sum0", "Top_Sum1", "Sum4", "Top_Sum3"});
    displayAsGraph("Elim step 2", 2);
        */
  }

  int n;
  int nh;
  Matrix A;
  Matrix Ah;
  Vector b;
  Vector bh;
  std::vector<std::string> varLabels;
  std::vector<std::string> rowLabels;
  int pivot;
};

int main()
{
  Kokkos::initialize();
  {
    Matrix A = randMat(4);
    Vector b = randVec(4);
    BMK bmk(A, b);
    //bmk.displayAsGraph("Initia0);
    bmk.noninteractiveEliminate();
  }
  Kokkos::finalize();
  return 0;
}

