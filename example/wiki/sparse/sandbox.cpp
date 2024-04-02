#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosSparse_Utils.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include "KokkosBlas.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_CooMatrix.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_coo2crs.hpp"
#include "KokkosGraph_Distance2Color.hpp"
#include <iostream>
#include <map>

using Scalar  = double;
using Ordinal = int;
using Offset  = int;
using Layout  = default_layout;
using Device  = Kokkos::DefaultExecutionSpace;
using KAT     = Kokkos::ArithTraits<Scalar>;

using Matrix = Kokkos::View<Scalar**, Kokkos::LayoutRight, Device>;
using CrsMatrix = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
using CrsGraph = typename CrsMatrix::StaticCrsGraphType;
using CooMatrix = KokkosSparse::CooMatrix<Scalar, Ordinal, Device, void, Offset>;
using Vector = Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>;
using IntVector = Kokkos::View<Ordinal*, Kokkos::LayoutLeft, Device>;
using MultiVector = Kokkos::View<Scalar**, Kokkos::LayoutRight, Device>;

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

// Fill v with normally distributed numbers (mean 0, stddev 1)
template<typename T>
void fillStandardNormal1D(const T& v)
{
  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> pool(rand() % 1234567);
  Kokkos::parallel_for(Kokkos::RangePolicy<typename T::execution_space>(0, v.extent(0)),
    KOKKOS_LAMBDA(int i)
      {
        auto randGen = pool.get_state();
        v(i) = randGen.normal();
        pool.free_state(randGen);
      });
}

// Fill v with normally distributed numbers (mean 0, stddev 1)
template<typename T>
void fillStandardNormal2D(const T& v)
{
  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> pool(rand() % 1234567);
  Kokkos::parallel_for(Kokkos::MDRangePolicy<typename T::execution_space, Kokkos::Rank<2>>({0, 0}, {v.extent(0), v.extent(1)}),
    KOKKOS_LAMBDA(int i, int j)
      {
        auto randGen = pool.get_state();
        v(i, j) = randGen.normal();
        pool.free_state(randGen);
      });
}

template<typename T>
void fillRand(const T& v)
{
  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(12345);
  Kokkos::fill_random(v, rand_pool, -1.0, 1.0);
}

Vector randVec(int n)
{
  Vector v("v", n);
  fillRand(v);
  return v;
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

Vector mysolve(const Matrix& A, const Vector& b)
{
  int n = b.extent(0);
  Vector x("x", n);
  return x;
}

// Hypersparse version of system consists of several parts
// - The matrix, as CRS
// - A mapping from hypersparse variables to original variables (many -> one)
// - The transposed graph of the hypersparse matrix
struct BMK
{
  BMK(CrsMatrix A_, Vector b_)
    : A(A_), b(b_)
  {
    auto At = KokkosSparse::Impl::transpose_matrix(A);
    int numVars = 0;
    int numRows = 0;
    // Build Ah as COO
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    std::vector<double> bvec;
    using Entry = std::pair<int, int>;
    using EntryVal = std::pair<int, double>;
    std::map<Entry, int> entryToVar;
    // First, create an unknown variable for each entry in A, and add equality constraints
    // It's easiest to do this by iterating down columns (rows of At)
    for(int c = 0; c < At.numRows(); c++)
    {
      // Record the previous free variable in the column
      int lastVar = -1;
      for(int j = At.graph.row_map(c); j < At.graph.row_map(c + 1); j++)
      {
        int r = At.graph.entries(j);
        int var = numVars++;
        entryToVar[Entry(r, c)] = var;
        if(lastVar == -1)
          lastVar = var;
        else {
          // Add constraint that var is equal to lastVar:
          // var - lastVar = 0
          int hrow = numRows++;
          rows.push_back(hrow);
          cols.push_back(lastVar);
          vals.push_back(1);
          rows.push_back(hrow);
          cols.push_back(var);
          vals.push_back(-1);
          bvec.push_back(0);
          lastVar = var;
        }
      }
    }
    // Next, for each row of the original matrix, express that
    // <A_row, x> = b_row using a binary tree reduction
    for(int r = 0; r < A.numRows(); r++)
    {
      double bval = b(r);
      std::vector<EntryVal> valuesToReduce;
      for(int j = A.graph.row_map(r); j < A.graph.row_map(r + 1); j++)
      {
        int c = A.graph.entries(j);
        // Get the free variable for which this entry is the coefficient
        int hvar = entryToVar[Entry(r, c)];
        double v = A.values(j);
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
          int sumvar = numVars++;
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
    std::cout << "Orig problem has " << A.numRows() << " unknowns and " << A.nnz() << " nonzeros.\n";
    std::cout << "Hypersparse problem has " << numVars << " unknowns and " << rows.size() << " nonzeros.\n";
    std::cout << "Number of rows should match unknowns: " << numRows << '\n';
    std::cout << "The system, as COO:\n";
    for(size_t i = 0; i < rows.size(); i++)
    {
      std::cout << "(" << rows[i] << ", " << cols[i] << ") = " << vals[i] << '\n';
    }
    nh = numVars;
    varMap = IntVector("varMap", numVars);
    // For each i, compute the ellipsoid coefficient in x_i^2.
    // Will use this to rescale each variable so that the x_i^2 coeffs are all 1.
    IntVector varDuplication("varDup", A.numRows());
    Vector varScaling("varScaling", nh);
    for(size_t i = 0; i < rows.size(); i++) {
      varScaling(cols[i]) += vals[i] * vals[i];
    }
    for(int i = 0; i < nh; i++)
      varScaling(i) = Kokkos::sqrt(varScaling(i));
    Kokkos::deep_copy(varMap, -1);
    varWeights = Vector("varWeights", nh);
    // Use entryToVar to populate varMap
    for(const auto& e : entryToVar)
    {
      int origVar = e.first.second;
      int newVar = e.second;
      varMap(newVar) = origVar;
      varDuplication(origVar)++;
    }
    for(int i = 0; i < nh; i++)
    {
      if(varMap(i) >= 0)
        varWeights(i) = varScaling(i) / varDuplication(varMap(i));
    }
    // Finally, rescale all hypersparse coeffs
    for(size_t i = 0; i < rows.size(); i++)
      vals[i] /= varScaling(cols[i]);
    // Everything ready to construct the persistent data structures
    Kokkos::View<int*> cooRows(rows.data(), rows.size());
    Kokkos::View<int*> cooCols(cols.data(), cols.size());
    Kokkos::View<double*> cooVals(vals.data(), vals.size());
    Ah = KokkosSparse::coo2crs(nh, nh, cooRows, cooCols, cooVals);
    transGraph = KokkosSparse::Impl::transpose_matrix(Ah).graph;
    bh = Vector("bh", nh);
    for(int i = 0; i < nh; i++)
      bh(i) = bvec[i];
    // Compute the bipartite row coloring of Ah
    {
      using Handle = KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, typename Device::execution_space, typename Device::memory_space, typename Device::memory_space>;
      Handle handle;
      // Use the default algorithm (chosen based on ExecSpace)
      handle.create_distance2_graph_coloring_handle(KokkosGraph::COLORING_D2_DEFAULT);
      // Run coloring
      KokkosGraph::Experimental::bipartite_color_rows(&handle, nh, nh, Ah.graph.row_map, Ah.graph.entries);
      // Get the colors array, and the number of colors used from the handle.
      colors = handle.get_distance2_graph_coloring_handle()->get_vertex_colors();
      numColors = handle.get_distance2_graph_coloring_handle()->get_num_colors();
      handle.destroy_distance2_graph_coloring_handle();
      std::cout << "** Colored hypersparse A with " << numColors << " colors.\n";
    }
  }

  // Solve the system (producing x vector in original variables)
  // Internally, the iterate xh corresponds to the hypersparse system
  Vector solve()
  {
    double initialNorm = KokkosBlas::nrm2(b);
    Vector xh("xh", nh);
    Vector delta("delta", nh);
    for(int iter = 0; iter < 10; iter++)
    {
      {
        // Determine the current iterate in the original system, and calculate rel. res. norm
        Vector xorig("xorig", b.extent(0));
        for(int i = 0; i < nh; i++)
        {
          int origVar = varMap(i);
          if(origVar >= 0)
          {
            // just average all hypersparse variables corresponding to the original variable
            // (varWeights takes into account the number of such variables, and the rescaling on each one)
            xorig(origVar) += xh(i) * varWeights(i);
          }
        }
        Vector residual("residual", b.extent(0));
        Kokkos::deep_copy(residual, b);
        KokkosSparse::spmv("N", 1.0, A, xorig, -1, residual);
        double currentNorm = KokkosBlas::nrm2(residual);
        printf("Iter %d: relative residual = %.3e\n", iter, currentNorm / initialNorm);
      }
      // xiter will be computed in this iteration based on bh.
      // Then xh will be updated with xiter, and bh will be updated to the residual.
      Vector xiter("xiter", nh);
      // Compute xiter initially as the center of the axis-aligned ellipsoid
      // (based only on the x_i^2 and x_i coefficients - ignoring x_i*x_j coefficients

    }
    //TODO
    return Vector();
  }

  CrsMatrix A;
  Vector b;
  int nh;
  CrsMatrix Ah;
  Vector bh;
  // If i is a hypersparse unknown, varMap(i) gives the corresponding
  // original unknown (or -1 if i is an intermediate sum variable)
  IntVector varMap;
  // If i is an unknown in the hypersparse system, and j is varMap(i),
  // then varWeights(i) * xh(i) gives the contribution to x(j).
  Vector varWeights;
  CrsGraph transGraph;
  IntVector colors;
  int numColors;
};

/*
void pcg(const RemoteVector& x, const LocalVector& b) {
  // TODO: make maxiter configurable
  const int maxiter = 100;
  // Initialize x to 0
  Kokkos::deep_copy(x.getLocalPart(), 0.0);
  // Initialize residual r = b - Ax_0 = b
  Kokkos::deep_copy(r, b);
  double initialResNorm = nrm2(r);
  // Initialize z = precond(r)
  jacobi(z, r);
  // Initialize p = z
  Kokkos::deep_copy(p.getLocalPart(), z);
  // Initialize r_dot_z
  double r_dot_z    = dot(z, r);
  double relResNorm = 1.0;
  for (int iter = 0; iter < maxiter; iter++) {
    // Apply linear operator PLPT to p, to get Ap
    if(iter != 0) RemoteSpace().fence();  // Fence because PLPT depends on remote entries of p
    PLPT(Ap, p);
    double p_dot_Ap = dot(p.getLocalPart(), Ap);
    if (p_dot_Ap <= 0.0) {
      std::cout << "Numerical breakdown: <p, Ap> = " << p_dot_Ap << "\n";
      throw std::runtime_error(
          "p_dot_Ap is not positive; operator is not positive definite");
    }
    double alpha = r_dot_z / p_dot_Ap;
    // Update x
    axpy(x.getLocalPart(), alpha, p.getLocalPart());
    // Update r
    axpy(r, -alpha, Ap);
    // Check if residual is small enough to terminate
    relResNorm = nrm2(r) / initialResNorm;
    if (relResNorm < tolerance) {
      std::cout << "PCG converged in " << iter+1 << " iters (relative tolerance " << relResNorm << " < " << tolerance << ")\n";
      return;
    }
    // Apply preconditioner
    jacobi(z, r);
    double new_r_dot_z = dot(z, r);
    double beta        = new_r_dot_z / r_dot_z;
    r_dot_z            = new_r_dot_z;
    // Update p
    axpby(p.getLocalPart(), beta, p.getLocalPart(), 1.0, z);
  }
  std::cout << "** WARNING: tolerance of " << tolerance
            << " not achieved after max iters (" << maxiter << ")\n";
  std::cout << "** Final relative residual norm: " << relResNorm << '\n';
}
*/

int main()
{
  Kokkos::initialize();
  {
    CrsMatrix A = KokkosSparse::Impl::read_kokkos_crst_matrix<CrsMatrix>("tiny.mtx");
    int n = A.numRows();
    Vector xgold = randVec(n);
    Vector b("b", n);
    KokkosSparse::spmv("N", 1.0, A, xgold, 0, b);
    std::cout << "Orig b: ";
    print1D(b);
    BMK bmk(A, b);
    std::cout << "H -> orig variable map:\n";
    print1D(bmk.varMap);
    std::cout << "bh:\n";
    print1D(bmk.bh);
    Vector x = bmk.solve();
  }
  Kokkos::finalize();
  return 0;
}

