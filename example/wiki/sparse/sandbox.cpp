#include "Kokkos_Core.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_IOUtils.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas1_nrm2_squared.hpp"
#include "KokkosBlas1_dot.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "KokkosBlas2_gemv.hpp"
#include "KokkosBlas3_gemm.hpp"
#include "KokkosBlas_gesv.hpp"

using Scalar  = default_scalar;
using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;
using Device  = Kokkos::DefaultExecutionSpace;
using KAT     = Kokkos::ArithTraits<Scalar>;

using Matrix = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
using Vector = Kokkos::View<Scalar*, Kokkos::LayoutLeft, Device>;
using MultiVector = Kokkos::View<Scalar**, Kokkos::LayoutLeft, Device>;
using PivotVector = Kokkos::View<int*, Kokkos::LayoutLeft, Device>;

// Ax - b
Vector residual(const Matrix& A, const Vector& x, const Vector& b)
{
  auto n = b.extent(0);
  Vector res("res", n);
  Kokkos::deep_copy(res, b);
  KokkosSparse::spmv("N", 1.0, A, x, -1.0, res);
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
  KokkosSparse::spmv("T", 1.0, A, res, 0.0, g);
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

//Reflect vector dir against plane with normal norm
Vector reflect(const Vector& norm, const Vector& dir)
{
  auto n = dir.extent(0);
  //Make sure that n is actually normalized
  Vector nnorm = normalize(norm);
  Scalar d = KokkosBlas::dot(nnorm, dir);
  Vector rdir("rdir", n);
  Kokkos::deep_copy(rdir, dir);
  KokkosBlas::axpby(-2 * d, nnorm, 1.0, rdir);
  return rdir;
}

//Use modified Gram-Schmidt to make the basis orthonormal.
//This might be important if some vectors are nearly linearly dependent.
MultiVector mgs(const MultiVector& basis)
{
  int k = basis.extent(1);
  for(int i = 0; i < k; i++)
  {
    auto icol = Kokkos::subview(basis, Kokkos::ALL(), i);
    auto normCol = normalize(icol);
    Kokkos::deep_copy(icol, normCol);
    for(int j = i + 1; j < k; j++)
    {
      auto jcol = Kokkos::subview(basis, Kokkos::ALL(), j);
      Scalar d = KokkosBlas::dot(icol, jcol);
      KokkosBlas::axpby(-d, icol, 1, jcol);
    }
  }
  return basis;
}

Vector iterate(const Matrix& A, const Vector& b)
{
  auto n = A.numRows();
  // Rank or dimension of the search subspace determined by bouncing vectors inside the contour
  const int rank = 4;
  MultiVector basis("subspace", n, rank);
  MultiVector basisImage("subspace (image via A)", n, rank);
  // The system matrix for solving the LSS.
  MultiVector lssA("least-squares A", rank, rank);
  // The LHS for the full-rank version of the LSS.
  Vector coeffs("coeffs", rank);
  // The RHS for the full-rank version of the LSS.
  Vector lssB("coeffs", rank);
  //The first column of the search subspace is the negative loss gradient from origin.
  //The remaining columns come from reflecting the previous search vector off the loss contour.
  //  (The normal to the loss countour is just given by the gradient at that point)
  Vector p("p", n);
  // All gradient evaluations will happen on the surface defined by: loss(A, x, b) == contourLoss
  Scalar contourLoss = loss(A, p, b);
  std::cout << "** Contour loss (eval. at origin): " << contourLoss << '\n';
  Vector search = gradient(A, p, b);
  Kokkos::deep_copy(Kokkos::subview(basis, Kokkos::ALL(), 0), search);
  for(int k = 1; k < rank; k++)
  {
    // Find the intersection of ray "p + t*search" with the contour
    Vector Asearch("Asearch", n);
    KokkosSparse::spmv("N", 1.0, A, search, 0.0, Asearch);
    Vector res = residual(A, p, b);
    Scalar Avnorm2 = KokkosBlas::nrm2_squared(Asearch);
    Scalar t = -2.0 * KokkosBlas::dot(Asearch, res) / Avnorm2;
    Vector newP("newsearch", n);
    Kokkos::deep_copy(newP, p);
    KokkosBlas::axpby(t, search, 1.0, newP);
    std::cout << "   Contour loss at bounce point " << k << ": " << loss(A, newP, b) << '\n';
    Vector grad = normalize(gradient(A, newP, b));
    // Bounce the old search direction off of gradient to get new search dir
    //Vector newSearch = reflect(grad, search);
    Vector newSearch = grad;
    KokkosBlas::scal(newSearch, -1.0, newSearch);
    // Sanity check reflect: input and output should be the same length
    std::cout << "Norms before/after bounce: " << KokkosBlas::nrm2(search) << "/" << KokkosBlas::nrm2(newSearch) << '\n';
    // Copy into the basis
    Kokkos::deep_copy(Kokkos::subview(basis, Kokkos::ALL(), k), newSearch);
    // and update p, search
    Kokkos::deep_copy(search, newSearch);
    Kokkos::deep_copy(p, newP);
  }
  // Now orthogonalize the search basis (may not be necessary, but reduces condition number of LSS)
  basis = mgs(basis);
  // Find the image of the basis (this is the LSS matrix)
  KokkosSparse::spmv("N", 1.0, A, basis, 0.0, basisImage);
  // Now solve the LSS: basisImage * coeffs = b
  // Do that by forming a full-rank linear system: basisImage^T * basisImage * coeffs = basisImage^T * b
  KokkosBlas::gemm("T", "N", 1.0, basisImage, basisImage, 0.0, lssA);
  KokkosBlas::gemv("T", 1.0, basisImage, b, 0.0, lssB);
  {
    //Directly solve that system
    //Note: gesv will overwrite lssA, but this is OK, we don't need it anymore
    //The solution will be placed into lssB (we don't need that anymore either)
    PivotVector ipiv("ipiv", rank);
    KokkosBlas::gesv(lssA, lssB, ipiv);
    Kokkos::deep_copy(coeffs, lssB);
  }
  Vector x("x", n);
  //Form the final x (approx) for this iteration, by taking a linear combo of basis using coeffs
  KokkosBlas::gemv("N", 1.0, basis, coeffs, 0.0, x);
  return x;
}

void solve(const Matrix& A, const Vector& b)
{
  int n = A.numRows();
  //Relative residual norm, where the
  //resnorm of initial guess (0 vector) is 1.0
  Scalar tol = 1e-11;
  Vector x("x", n);
  Scalar initResNorm = Kokkos::sqrt(loss(A, x, b));
  std::cout << "Iter 0: scaled res norm: 1\n";
  for(int iter = 1;; iter++)
  {
    // At each iteration, solve for the residual
    Vector res = residual(A, x, b);
    Vector update = iterate(A, res);
    // and update x
    KokkosBlas::axpby(-1.0, update, 1.0, x);
    Scalar relResNorm = Kokkos::sqrt(loss(A, x, b)) / initResNorm;
    std::cout << "Iter " << iter << ": scaled res norm: " << relResNorm << "\n";
    if(relResNorm <= tol)
    {
      std::cout << "Converged to desired tolerance of " << tol << ", done.\n";
      break;
    }
  }
}

int main(int argc, const char** argv)
{
  if(argc != 2)
  {
    std::cout << "Provide matrix (MatrixMarket format)\n";
    return 0;
  }

  Kokkos::initialize();
  {
    //Set up problem
    std::cout << "Reading problem matrix from \"" << argv[1] << "\"\n";
    Matrix A = KokkosSparse::Impl::read_kokkos_crst_matrix<Matrix>(argv[1]);
    //Create unit-length random RHS
    Kokkos::Random_XorShift64_Pool<typename Device::execution_space> rand_pool(12345);
    auto n = A.numRows();
    Vector b("b", n);
    Kokkos::fill_random(b, rand_pool, -10.0, 10.0);
    b = normalize(b);
    solve(A, b);
  }
  Kokkos::finalize();
  return 0;
}

