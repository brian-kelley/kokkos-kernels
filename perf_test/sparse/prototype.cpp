#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>

using std::cout;
using Kokkos::subview;

using Matrix = Kokkos::View<double**, Kokkos::HostSpace>;
using Vector = Kokkos::View<double*, Kokkos::HostSpace>;
using ConstVector = Kokkos::View<const double*, Kokkos::HostSpace>;
constexpr int n = 64;

template<typename M, typename CV, typename V>
void check(const M& A, const CV& b, const V& x)
{
  double err = 0;
  assert(A.extent(0) == A.extent(1));
  int an = A.extent(0);
  for(int i = 0; i < an; i++)
  {
    double resid = b(i);
    for(int j = 0; j < an; j++)
    {
      resid -= A(i, j) * x(j);
    }
    err += resid * resid;
  }
  std::cout << "Error: " << err << '\n';
}

void runSerial(const Matrix& A, const ConstVector& b)
{
  Vector x("x", n);
  for(int i = 0; i < n; i++)
  {
    double resid = b(i);
    for(int j = 0; j < i; j++)
    {
      resid -= A(i, j) * x(j);
    }
    x(i) = resid / A(i, i);
  }
  check(A, b, x);
}

void runHierarchical(const Matrix& A, const ConstVector& b)
{
  Vector x("x", n);
  for(int i = 0; i < n; i++)
  {
    x(i) = b(i) / A(i, i);
  }
  for(int bs = 1; bs < n; bs *= 2)
  {
    //cout << "Starting blocksize = " << bs << '\n';
    for(int begin = 0; begin < n; begin += 2 * bs)
    {
      int mid = begin + bs;
      int end = mid + bs;
      /*
      cout << "Updating with block: A(" << mid << ":" << end << ", " << begin << ":" << mid << ")\n";
      cout << "  Checking that incoming x(begin:mid) solves system...\n";
      cout << "  ";
      check(
          subview(A, Kokkos::make_pair(begin, mid), Kokkos::make_pair(begin, mid)),
          subview(b, Kokkos::make_pair(begin, mid)),
          subview(x, Kokkos::make_pair(begin, mid)));
      */
      //update: x(mid..end) -= D^-1(mid:end) * A(mid:end, begin:mid) * x(begin:mid)
      Vector xUpdate("update", bs);
      for(int i = mid; i < end; i++)
      {
        double sum = 0;
        for(int j = begin; j < mid; j++)
        {
          sum += A(i, j) * x(j);
        }
        for(int j = mid; j < i; j++)
        {
          sum += A(i, j) * xUpdate(j - mid);
        }
        //std::cout << "  Updating x(" << i << ")\n";
        xUpdate(i - mid) = -sum / A(i, i);
        x(i) -= sum / A(i, i);
      }
    }
  }
  check(A, b, x);
}

int main()
{
  Kokkos::initialize();
  {
    //Create random system
    Matrix A("A", n, n);
    Vector b("b", n);
    for(int i = 0; i < n; i++)
    {
      b(i) = (4.0 * rand()) / RAND_MAX;
      for(int j = 0; j <= i; j++)
      {
        if(i == j)
          A(i, j) = -n * (0.5 + (0.5 * rand()) / RAND_MAX);
        else
          A(i, j) = (1.0 * rand()) / RAND_MAX;
      }
    }
    std::cout << "Running serial solve:\n";
    runSerial(A, b);
    std::cout << '\n';
    std::cout << "Running hierarchical solve:\n";
    runHierarchical(A, b);
    std::cout << '\n';
  }
  Kokkos::finalize();
  return 0;
}

