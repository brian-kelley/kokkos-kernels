#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>

using std::cout;
using Kokkos::subview;

using Matrix = Kokkos::View<double**, Kokkos::HostSpace>;
using Vector = Kokkos::View<double*, Kokkos::HostSpace>;
using ConstVector = Kokkos::View<const double*, Kokkos::HostSpace>;
constexpr int n = 4;

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
    //std::cout << "    Row " << i << " resid: " << fabs(resid) << '\n';
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

void runScan(const Matrix& A, const ConstVector& b)
{
  Vector x("y", n);
  Vector change("change", n);
  Matrix B("B", n, n);
  for(int i = 0; i < n; i++)
  {
    double dinv = 1.0 / A(i, i);
    for(int j = 0; j <= i; j++)
    {
      B(i, j) = A(i, j) * dinv;
    }
    x(i) = b(i) * dinv;
  }
  //Up-sweeps
  for(int bs = 1; bs < n; bs *= 2)
  {
    std::cout << "Starting step " << 1 + log2(bs) << " of up-sweep\n";
    int bs2 = 2 * bs;
    //for each block...
    for(int begin = 0; begin < n; begin += bs2)
    {
      int mid = begin + bs;
      int end = mid + bs;
      //update x(mid:end) using -B * x(begin:mid)
      for(int i = mid; i < end; i++)
      {
        double sum = 0;
        for(int j = begin; j < mid; j++)
        {
          sum += B(i, j) * x(j);
        }
        std::cout << "  Updated x(" << i << ")\n";
        x(i) -= sum;
        change(i) = -sum;
      }
    }
  }
  //Save current values before down-sweeps begin
  //Down-sweeps
  for(int bs = n / 4; bs; bs /= 2)
  {
    std::cout << "Starting step " << -1 + log2(n / bs) << " of down-sweep\n";
    int bs2 = 2 * bs;
    //for each block...
    for(int begin = 0; begin < n; begin += bs2)
    {
      int mid = begin + bs;
      int end = mid + bs;
      //update x(mid:end) using -B * (x-update)(begin:mid)
      for(int i = mid; i < end; i++)
      {
        double sum = 0;
        for(int j = begin; j < mid; j++)
        {
          double chg = change(j);
          //std::cout << "x(" << j << ") has changed by " << change << '\n';
          sum += B(i, j) * chg;
        }
        std::cout << "    Updating x(" << i << ")\n";
        x(i) -= sum;
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
    runScan(A, b);
    std::cout << '\n';
  }
  Kokkos::finalize();
  return 0;
}

