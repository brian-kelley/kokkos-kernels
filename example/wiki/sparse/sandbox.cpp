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
using Matrix = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device>;
using Vector = Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>;
using MultiVector = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device>;

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
Vector residual(const Matrix& A, const Vector& x, const Vector& b)
{
  auto n = b.extent(0);
  Vector res("res", n);
  Kokkos::deep_copy(res, b);
  KokkosBlas::gemv("N", 1.0, A, x, -1.0, res);
  return res;
}

// loss = ||residual||^2
// This is objective function to be minimized
Scalar loss(const Matrix& A, const Vector& x, const Vector& b)
{
  return KokkosBlas::nrm2_squared(residual(A, x, b));
}

// grad(loss) = 2A^T * (Ax-b)
Vector gradient(const Matrix& A, const Vector& x, const Vector& b)
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

struct Solver
{
  Solver(const Matrix& A, const Vector& b)
  {
    n = A.extent(0);
    //First, choose the two dimensions (d1, d2) to search.
    //Pick those with the smallest magnitude loss gradient at the origin
    d1 = 0;
    d2 = 0;
    {
      Vector x("x", n);
      Vector g = gradient(A, x, b);
      KokkosBlas::abs(g, g);
      for(int i = 0; i < n; i++)
      {
        if(g(d1) > g(i))
          d1 = i;
      }
      //Make sure d2 can never be the same as d1
      if(d1 == 0)
        d2++;
      for(int i = 0; i < n; i++)
      {
        if(g(d2) > g(i) && i != d1)
          d2 = i;
      }
      if(d1 > d2)
        std::swap(d1, d2);
      assert(d1 != d2);
    }
    std::cout << "Using coordinate plane x_" << d1 << " * x_" << d2 << '\n';
    //Start computing vectors and quantities in terms of these x values.
    //Represent vectors as 3-column multivectors with x_d1, x_d2, and 1 as coefficients
    
    // b - Ax
    bmAx = MultiVector("Ax", n, 3);
    //Since x is just a sum of 2 elementary vectors, this is just selecting two columns of A
    for(int i = 0; i < n; i++)
    {
      bmAx(i, 0) = -A(i, d1);
      bmAx(i, 1) = -A(i, d2);
      bmAx(i, 2) = b(i);
    }
    // y = A'(b - Ax)
    // Compute by applying A' to each column with a gemm
    y = MultiVector("y", n, 3);
    KokkosBlas::gemm("T", "N", 1.0, A, bmAx, 0.0, y);
    // Ay
    Ay = MultiVector("Ay", n, 3);
    KokkosBlas::gemm("N", "N", 1.0, A, y, 0.0, Ay);
    // tn, td (t = tn/td). Each consists of 6 terms when fully reduced.
    // Terms are: d1^2, d2^2, d1d2, d1, d2, 1
    tn = Vector("tn", 6);
    td = Vector("td", 6);
    bmAx2 = Vector("bmAx2", 6);
    {
      Matrix tmp("tmp", 3, 3);
      //First, compute tn as <b-Ax, Ay>
      KokkosBlas::gemm("T", "N", 1.0, bmAx, Ay, 0.0, tmp);
      tn(0) = tmp(0, 0);
      tn(1) = tmp(1, 1);
      tn(2) = tmp(0, 1) + tmp(1, 0);
      tn(3) = tmp(0, 2) + tmp(2, 0);
      tn(4) = tmp(1, 2) + tmp(2, 1);
      tn(5) = tmp(2, 2);
      //Then compute td as <Ay, Ay>
      KokkosBlas::gemm("T", "N", 1.0, Ay, Ay, 0.0, tmp);
      td(0) = tmp(0, 0);
      td(1) = tmp(1, 1);
      td(2) = tmp(0, 1) + tmp(1, 0);
      td(3) = tmp(0, 2) + tmp(2, 0);
      td(4) = tmp(1, 2) + tmp(2, 1);
      td(5) = tmp(2, 2);
      //Finally, bmAx2 is <b-Ax, b-Ax>
      KokkosBlas::gemm("T", "N", 1.0, bmAx, bmAx, 0.0, tmp);
      bmAx2(0) = tmp(0, 0);
      bmAx2(1) = tmp(1, 1);
      bmAx2(2) = tmp(0, 1) + tmp(1, 0);
      bmAx2(3) = tmp(0, 2) + tmp(2, 0);
      bmAx2(4) = tmp(1, 2) + tmp(2, 1);
      bmAx2(5) = tmp(2, 2);
    }
    std::cout << "Done constructing solver. Let (x, y) = (x_" << d1 << ", x_" << d2 << ")\n";
    std::cout << "Let g(x,y) = c - a*a/b\n";
    std::cout << "a = " << tn(0) << "xx + " << tn(1) << "yy + " << tn(2) << "xy + " << tn(3) << "x + " << tn(4) << "y + " << tn(5) << '\n';
    std::cout << "b = " << td(0) << "xx + " << td(1) << "yy + " << td(2) << "xy + " << td(3) << "x + " << td(4) << "y + " << td(5) << '\n';
    std::cout << "c = " << bmAx2(0) << "xx + " << bmAx2(1) << "yy + " << bmAx2(2) << "xy + " << bmAx2(3) << "x + " << bmAx2(4) << "y + " << bmAx2(5) << '\n';
  }

  Scalar eval6Term(Scalar xd1, Scalar xd2, const Vector& coef)
  {
    return
      coef(0) * xd1 * xd1 +
      coef(1) * xd2 * xd2 +
      coef(2) * xd1 * xd2 +
      coef(3) * xd1 +
      coef(4) * xd2 +
      coef(5);
  }

  // Function to minimize: g(x_d1, x_d2)
  // gives the squared residual norm after one step of gradient descent with line search, starting from x.
  Scalar g(Scalar xd1, Scalar xd2)
  {
    Scalar evalTN = eval6Term(xd1, xd2, tn);
    Scalar evalTD = eval6Term(xd1, xd2, td);
    Scalar evalBMAX2 = eval6Term(xd1, xd2, bmAx2);
    return evalBMAX2 - evalTN * evalTN / evalTD;
  }
  
  Vector grad_g(Scalar xd1, Scalar xd2)
  {
    Vector grad("grad", 2);
    return grad;
  }

  Matrix hess_g(Scalar xd1, Scalar xd2)
  {
    Matrix hess("hess", 2, 2);
    return hess;
  }

  int n;
  int d1;
  int d2;
  MultiVector bmAx;
  MultiVector y;
  MultiVector Ay;
  // The following are scalar expressions in terms of x_d1, x_d2
  // There are 6 terms: d1^2, d2^2, d1d2, d1, d2, 1
  Vector tn;
  Vector td;
  Vector bmAx2;
};

void solve(const Matrix& A, const Vector& b)
{
  Solver s(A, b);
  Scalar test_x1 = 0.383;
  Scalar test_x2 = -0.024;
  std::cout << "Solver using plane x_" << s.d1 << ", x_" << s.d2 << '\n';
  std::cout << "Solver says g(" << test_x1 << ", " << test_x2 << ") = " << s.g(test_x1, test_x2) << '\n';
  // Now form the full, explicit vector there
  Vector x("x", s.n);
  x(s.d1) = test_x1;
  x(s.d2) = test_x2;
  // Compute gradient
  Vector grad = gradient(A, x, b);
  // Compute optimal step size from x
  Vector res = residual(A, x, b);
  Vector Agrad("Agrad", s.n);
  KokkosBlas::gemv("N", 1.0, A, grad, 0.0, Agrad);
  Scalar t = -KokkosBlas::dot(res, Agrad) / KokkosBlas::nrm2_squared(Agrad);
  Vector newX("newX", s.n);
  Kokkos::deep_copy(newX, x);
  KokkosBlas::axpy(t, grad, newX);
  Scalar newResNrm = KokkosBlas::nrm2_squared(residual(A, newX, b));
  std::cout << "Explicit calc says grad.desc. from x gives " << newResNrm << '\n';
}

int main(int argc, const char** argv)
{
  /*
  if(argc != 2)
  {
    std::cout << "Provide matrix (MatrixMarket format)\n";
    return 0;
  }
  */

  Kokkos::initialize();
  {
    //Set up problem
    /*
    std::cout << "Reading problem matrix from \"" << argv[1] << "\"\n";
    Matrix A = KokkosSparse::Impl::read_kokkos_crst_matrix<Matrix>(argv[1]);
    auto n = A.numRows();
    */
    int n = 10;
    Matrix A("A", n, n);
    fillRand(A);
    for(int i = 0; i < n; i++)
    {
      A(i, i) = KAT::abs(A(i, i));
      A(i, i) += 1.0;
    }
    //Create unit-length random RHS
    Vector b = normalize(randVec(n));
    solve(A, b);
  }
  Kokkos::finalize();
  return 0;
}

