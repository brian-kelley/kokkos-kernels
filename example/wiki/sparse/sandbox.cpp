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

// Fill v with normally distributed numbers (mean 0, stddev 1)
template<typename T>
void fillRademacher(const T& v)
{
  Kokkos::Random_XorShift64_Pool<typename Device::execution_space> pool(rand() % 1234567);
  Kokkos::parallel_for(Kokkos::MDRangePolicy<typename T::execution_space, Kokkos::Rank<2>>({0, 0}, {v.extent(0), v.extent(1)}),
    KOKKOS_LAMBDA(int i, int j)
      {
        auto randGen = pool.get_state();
        double val = randGen.normal();
        if(val < 0)
          v(i, j) = -1;
        else
          v(i, j) = 1;
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

//Scale v to be unit length.
void normalizeInPlace(const Vector& v)
{
  auto norm = KokkosBlas::nrm2(v);
  KokkosBlas::scal(v, 1.0 / norm, v);
}

// Compute a Householder reflection vector v, such that
// if H = (I - (1/tau)*vv^T), then Hx = c*e_k. Assumes that input element x_k is nonzero.
template<typename T>
Vector householder(const T& x, int k, double& tau, double& c)
{
  c = KokkosBlas::nrm2(x);
  Vector v("HH v", x.extent(0));
  //double sk = (x(k + 1) < 0.0) ? -1.0 : 1.0;
  double sk = 1.0;
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
  double alpha = -d / tau;
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

//Replace M by an orthonormal basis for its rows using modified Gram-Schmidt.
template<typename T>
void orthonormalize(const T& M)
{
  int m = M.extent(0);
  for(int i = 0; i < m; i++)
  {
    auto irow = Kokkos::subview(M, i, Kokkos::ALL());
    normalizeInPlace(irow);
    //Now that row i is unit length, orthogonalize the subsequent rows against it
    for(int j = i + 1; j < m; j++)
    {
      auto jrow = Kokkos::subview(M, j, Kokkos::ALL());
      double d = KokkosBlas::dot(irow, jrow);
      KokkosBlas::axpy(-d, irow, jrow);
    }
  }
}

// x1 and x2 are the projections of x into two (possibly non-orthogonal) subspaces.
// Returns the projection of x into the union of these subspaces.
Vector join(const Vector& x, const Vector& x1, const Vector& x2)
{
  int n = x.extent(0);
  Vector x1norm = normalize(x1);
  Vector x2norm = normalize(x2);
  double d = KokkosBlas::dot(x1norm, x2norm);
  Vector x2normOrthog("asdf", n);
  Kokkos::deep_copy(x2normOrthog, x2norm);
  KokkosBlas::axpy(-d, x1norm, x2normOrthog);
  x2normOrthog = normalize(x2normOrthog);
  Vector xproj("xproj", n);
  double d1 = KokkosBlas::dot(x, x1norm);
  double d2 = KokkosBlas::dot(x, x2normOrthog);
  KokkosBlas::axpy(d1, x1norm, xproj);
  KokkosBlas::axpy(d2, x2normOrthog, xproj);
  return xproj;
}

/*
void testHH()
{
  Matrix A("A", 5, 5);
  fillRand(A);
  std::cout << "Matrix A:\n";
  print2D(A);
  double tau, c;
  auto v = householder(Kokkos::subview(A, Kokkos::ALL(), 1), 4, tau, c);
  applyHouseholderLeft(A, v, tau);
  std::cout << "Matrix A, after eliminating the column 1 except element 4:\n";
  print2D(A);
  v = householder(Kokkos::subview(A, 3, Kokkos::ALL()), 2, tau, c);
  applyHouseholderRight(A, v, tau);
  std::cout << "Now, Matrix A, after eliminating the row 3 except element 2:\n";
  print2D(A);
  std::cout << "Forming H explicitly.\n";
  Matrix H = identity(5);
  applyHouseholderRight(H, v, tau);
  print2D(H);
  std::cout << "H^2:\n";
  Matrix H2("H*H", 5, 5);
  KokkosBlas::gemm("N", "N", 1.0, H, H, 0.0, H2);
  print2D(H2);
}
*/

int main()
{
  Kokkos::initialize();
  {
    /*
    Vector v1("v1", 3);
    v1(0) = 1;
    v1(1) = 2;
    v1(2) = 3;
    Vector v2("v2", 3);
    v2(0) = 1;
    v2(1) = 1;
    v2(2) = 0;
    Vector x("x", 3);
    x(0) = 1;
    x(1) = 0;
    x(2) = 0;
    Vector joined = join(x, v1, v2);
    print1D(joined);
    */


    int n = 17;
    Matrix A("A", n, n);
    fillRand(A);
    for(int i = 0; i < n; i++)
      normalizeInPlace(Kokkos::subview(A, i, Kokkos::ALL()));
    Vector b = randVec(n);
    //make sure b(0) is not tiny
    b(0) = Kokkos::abs(b(0));
    b(0) += 0.5;
    b = normalize(b);
    std::cout << "System matrix A:\n";
    print2D(A);
    std::cout << "\n b (RHS):\n";
    print1D(b);

    // Find H that maps b to e0
    double tau, c;
    auto v = householder(b, 0, tau, c);
    Vector hb("hb", n);
    Kokkos::deep_copy(hb, b);
    applyHouseholder1D(hb, v, tau);
    Matrix HA("HA", n, n);
    Kokkos::deep_copy(HA, A);
    applyHouseholderLeft(HA, v, tau);
    std::cout << "Equivalent, reflected system:\n";
    print2D(HA);
    std::cout << "\n and:\n";
    print1D(hb);

    Matrix HAOrthog("HAorthog", n-1, n);
    Kokkos::deep_copy(HAOrthog, Kokkos::subview(HA, Kokkos::make_pair(1, n), Kokkos::ALL()));
    orthonormalize(HAOrthog);

    //Create a copy of first row of HA
    //This will be projected into the span of the remaining rows
    Vector x = normalize(Kokkos::subview(HA, 0, Kokkos::ALL()));

    Vector xcoeffs("asdf", n - 1);
    KokkosBlas::gemv("N", 1.0, HAOrthog, x, 0, xcoeffs);
    Vector correctXproj("asdf", n);
    KokkosBlas::gemv("T", 1.0, HAOrthog, xcoeffs, 0, correctXproj);
    correctXproj = normalize(correctXproj);
    std::cout << "Correct xproj:\n";
    print1D(correctXproj);

    Matrix projections("projections", n-1, n);
    Kokkos::deep_copy(projections, Kokkos::subview(HA, Kokkos::make_pair(1, n), Kokkos::ALL()));
    for(int gap = 1; gap < n; gap *= 2)
    {
      std::cout << "Starting projection/reduction: gap = " << gap << '\n';
      for(int joinDst = 0; joinDst < n; joinDst += gap * 2)
      {
        int joinSrc = joinDst + gap;
        if(joinSrc >= projections.extent(0))
          continue;
        std::cout << "  Joining projections in row " << joinSrc << " into row " << joinDst << "\n";
        Vector proj1 = Kokkos::subview(projections, joinSrc, Kokkos::ALL());
        Vector proj2 = Kokkos::subview(projections, joinDst, Kokkos::ALL());
        Vector newProj = join(x, proj1, proj2);
        //Store the combined projection into joinDst
        Kokkos::deep_copy(Kokkos::subview(projections, joinDst, Kokkos::ALL()), newProj);
      }
    }
    Vector finalProj = normalize(Kokkos::subview(projections, 0, Kokkos::ALL()));
    //Now orthogonalize x against finalProj
    double d = KokkosBlas::dot(x, finalProj);
    Vector xortho("xortho", n);
    Kokkos::deep_copy(xortho, x);
    KokkosBlas::axpy(-d, finalProj, xortho);
    xortho = normalize(xortho);
    Vector res("res", n);
    KokkosBlas::gemv("N", 1.0, HA, xortho, 0, res);
    std::cout << "X, projected into span of rows except first:\n";
    print1D(finalProj);
    std::cout << "X, orthogonalized against span of rows except first:\n";
    print1D(xortho);
    std::cout << "HA * xortho (should be multiple of e0):\n";
    print1D(res);
  }
  Kokkos::finalize();
  return 0;
}

