#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include "KokkosBlas.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_CooMatrix.hpp"

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
// - The transposed graph of the matrix
void assembleHypersparse(CrsMatrix A, Vector b, CrsMatrix& Ah, Vector& bh, IntVector& varMap, CrsGraph& transGraph)
{
  auto At = KokkosSparse::Impl::transpose_matrix(A);
  int numVars = 0;
  // Build Ah as COO
  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<double> vals;
  // First, create an unknown variable for each entry in A, and add equality constraints
  // It's easiest to do this by iterating down columns (rows of At)
  for(int c = 0; c < At.numRows(); c++)
  {
    for(int j = At.graph.row_map(c); j < At.graph.row_map(c + 1); j++)
    {
      int r = 
    }
  }
}

void pcg(const RemoteVector& x, const LocalVector& b) {
  int myRank = getMyRank();
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
      if (myRank == 0)
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
    /*
    if (myRank == 0)
      std::cout << "Iter " << iter
                << " relative residual norm: " << relResNorm << '\n';
    */
    if (relResNorm < tolerance) {
      if(myRank == 0) std::cout << "PCG converged in " << iter+1 << " iters (relative tolerance " << relResNorm << " < " << tolerance << ")\n";
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
  if (myRank == 0) {
    std::cout << "** WARNING: tolerance of " << tolerance
              << " not achieved after max iters (" << maxiter << ")\n";
    std::cout << "** Final relative residual norm: " << relResNorm << '\n';
  }
}

int main()
{
  Kokkos::initialize();
  {
    int n = 10;
    Matrix Amat("A", n, n);
    fillRand(Amat);
    std::cout << "A matrix:\n";
    print2D(Amat);
    Vector xgold = randVec(n);
    Vector b("b", n);
    KokkosBlas::gemv("N", 1.0, Amat, xgold, 0, b);
    std::cout << "b vector:\n";
    print1D(b);
  }
  Kokkos::finalize();
  return 0;
}

