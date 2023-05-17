#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosBlas.hpp"

using Scalar  = default_scalar;
using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;
using Device  = Kokkos::DefaultExecutionSpace;
using KAT     = Kokkos::ArithTraits<Scalar>;

//using Matrix = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
using Matrix = Kokkos::View<Scalar**, Kokkos::LayoutRight, Device>;
using Vector = Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>;
using MultiVector = Kokkos::View<Scalar**, Kokkos::LayoutRight, Device>;

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

// loss = ||residual||^2
// This is objective function to be minimized
template<typename AT>
Scalar loss(const AT& A, const Vector& x, const Vector& b)
{
  return KokkosBlas::nrm2_squared(residual(A, x, b));
}

// grad(loss) = 2A^T * (Ax-b)
template<typename AT>
Vector gradient(const AT& A, const Vector& x, const Vector& b)
{
  auto n = b.extent(0);
  Vector res = residual(A, x, b);
  Vector g("g", n);
  KokkosBlas::gemv("T", 1.0, A, res, 0.0, g);
  KokkosBlas::scal(g, 2.0, g);
  return g;
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

// Compute a Householder reflection vector v, such that
// if H = (I - (1/tau)*vv^T), then Hx = c*e_k. Assumes that input element x_k is nonzero.
template<typename T>
Vector householder(const T& x, int k, double& tau, double& c)
{
  c = KokkosBlas::nrm2(x);
  Vector v("HH v", x.extent(0));
  double sk = (x(k) < 0.0) ? -1.0 : 1.0;
  Kokkos::deep_copy(v, x);
  v(k) += sk * c;
  tau = KokkosBlas::nrm2_squared(v) / 2;
  return v;
}

//Apply Householder reflector to a vector x.
//x may be a row or column vector, corresponding to application of H to the right or left.
//But the computation is the same in both cases.
template<typename T>
void applyHouseholder1D(const T& x, const Vector& v, double tau)
{
  static_assert(T::rank == 1, "Rank-1 only");
  double d = KokkosBlas::dot(x, v);
  double alpha = d / tau;
  KokkosBlas::axpy(alpha, v, x);
}

//Apply Householder reflector on the left (to the columns) of x.
//x must be 2D.
template<typename T>
void applyHouseholderLeft(const T& x, const Vector& v, double tau)
{
  int n = x.extent(1);
  for(int i = 0; i < n; i++)
  {
    auto xcol = Kokkos::subview(x, Kokkos::ALL(), i);
    applyHouseholder1D(xcol, v, tau);
  }
}

//Apply Householder reflector on the right (to the rows) of x.
//x must be 2D.
template<typename T>
void applyHouseholderRight(const T& x, const Vector& v, double tau)
{
  int m = x.extent(0);
  for(int i = 0; i < m; i++)
  {
    auto xrow = Kokkos::subview(x, i, Kokkos::ALL());
    applyHouseholder1D(xrow, v, tau);
  }
}

int main()
{
  Kokkos::initialize();
  {
    Matrix A("A", 5, 5);
    fillRand(A);
    std::cout << "Matrix A:\n";
    print2D(A);
    double tau, c;
    auto v = householder(Kokkos::subview(A, Kokkos::ALL(), 1), 1, tau, c);
    applyHouseholderLeft(A, v, tau);
    std::cout << "Matrix A, after eliminating the column 1 except element 1:\n";
    print2D(A);

    //Set up problem
    /*
    std::cout << "Reading problem matrix from \"" << argv[1] << "\"\n";
    Matrix A = KokkosSparse::Impl::read_kokkos_crst_matrix<Matrix>(argv[1]);
    auto n = A.numRows();
    */
    /*
    int n = 4;
    Matrix A("A", n, n);
    fillRand(A);
    for(int i = 0; i < n; i++)
    {
      A(i, i) = KAT::abs(A(i, i));
      A(i, i) += 1.0;
    }
    //Create unit-length random RHS
    Vector b = normalize(randVec(n));
    std::cout << "*** System matrix: ***\n\n";
    for(int i = 0; i < n; i++)
      KokkosKernels::Impl::print_1Dview(Kokkos::subview(A, i, Kokkos::ALL()));
    std::cout << "\n\n*** System RHS: ***\n\n";
    KokkosKernels::Impl::print_1Dview(b);
    std::cout << "\n***\n";
    solve(A, b);
    */
  }
  Kokkos::finalize();
  return 0;
}

