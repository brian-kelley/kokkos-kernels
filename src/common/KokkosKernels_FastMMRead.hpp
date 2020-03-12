/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef _KOKKOSKERNELS_FASTMMREAD_H_
#define _KOKKOSKERNELS_FASTMMREAD_H_

#include "KokkosKernels_Sorting.hpp"
#include "KokkosKernels_Utils.hpp"
#include <type_traits>
#include <cstdio>
#include <string>

namespace KokkosKernels{
namespace Impl{

/** Read a MatrixMarket file into a KokkosSparse::CrsMatrix.
 *  All parsing and symmetrization is performed on device,
 *  and the read from disk is done in a single fread() call.
 *
 *  For very large matrices (CRS format uses >1/3 of device memory),
 *  this function may not be able to allocate some views and Kokkos will throw an
 *  exception - in this case, just fall back to the read_mtx() that runs on host.
 *
 *  Like read_mtx(), the result always has sorted rows.
 */
template <typename crsMat_t>
int read_mtx_device(const char *fileName);

template<typename memory_space>
struct LineOffsetsFunctor
{
  using OffsetView = Kokkos::View<size_t*, memory_space>;
  using CharView = Kokkos::View<char*, memory_space>;

  LineOffsetsFunctor(const CharView& text_, const OffsetView& offsets_)
    : text(text_), offsets(offsets_) {}

  //i is the byte index into file
  //if finalPass, lline is the line in which byte i appears
  KOKKOS_INLINE_FUNCTION void operator()(size_t i, size_t& lline, bool finalPass) const
  {
    if(i == 0 || (i > 0 && text[i - 1] == '\n'))
    {
      //a line begins at this character
      if(finalPass)
        offsets[lline] = i;
      lline++;
    }
  }

  CharView text;
  OffsetView offsets;
};

struct DeviceParsing
{
  KOKKOS_INLINE_FUNCTION static long parseLong(char*& str)
  {
    bool minus = false;
    if(*str == '-')
    {
      minus = true;
      str++;
    }
    else if(*str == '+')
      str++;
    long val = 0;
    while(*str >= '0' && *str <= '9')
    {
      val *= 10;
      val += (*str - '0');
      str++;
    }
    if(minus)
      return -val;
    return val;
  }

  KOKKOS_INLINE_FUNCTION static double parseDouble(char*& str)
  {
    bool minus = false;
    if(*str == '-')
    {
      minus = true;
      str++;
    }
    //parse mantissa until '.' or 'E'/'e'
    double val = 0;
    while(*str != '.' && *str != 'e' && *str != 'E')
    {
      val *= 10;
      val += (double) (*str - '0');
      str++;
    }
    if(*str == '.')
    {
      str++;
      constexpr double invTen = 1.0 / 10;
      double placeval = invTen;
      while(*str != '.' && *str != 'e' && *str != 'E')
      {
        val += placeval * (*str - '0');
        placeval *= invTen;
        str++;
      }
    }
    if(*str == 'e' || *str == 'E')
    {
      str++;
      val *= Kokkos::ArithTraits<double>::pow(10.0, parseLong(str));
    }
    if(minus)
      return -val;
    return val;
  }

  //Float and double
  template<typename Scalar, typename std::enable_if<std::is_floating_point<Scalar>::value>::type* = nullptr>
  KOKKOS_INLINE_FUNCTION static Scalar parseScalar(char*& str)
  {
    return parseDouble(str);
  }

  //Integers
  template<typename Scalar, typename std::enable_if<std::is_integral<Scalar>::value>::type* = nullptr>
  KOKKOS_INLINE_FUNCTION static Scalar parseScalar(char*& str)
  {
    return parseLong(str);
  }

  //Kokkos::complex, either float or double
  template<typename Scalar, typename std::enable_if<
    std::is_same<Scalar, Kokkos::complex<double>>::value ||
    std::is_same<Scalar, Kokkos::complex<float>>::value>::type* = nullptr>
  KOKKOS_INLINE_FUNCTION static Scalar parseScalar(char*& str)
  {
    double real = parseDouble(str);
    while(*str == ' ' || *str == '\t')
      str++;
    double imag = parseDouble(str);
    return Scalar(real, imag);
  }
};

template<typename Rowmap, typename memory_space>
struct CountRowEntriesFunctor
{
  using OffsetView = Kokkos::View<size_t*, memory_space>;
  using CharView = Kokkos::View<char*, memory_space>;

  CountRowEntriesFunctor(const CharView& text_, const OffsetView& offsets_, const Rowmap& rowCounts_)
    : text(text_), offsets(offsets_), rowCounts(rowCounts_) {}

  //i is the byte index into file
  //if finalPass, lline is the line in which byte i appears
  KOKKOS_INLINE_FUNCTION void operator()(size_t line) const
  {
    size_t offset = offsets(line);
    //Parse the row
    long r = DeviceParsing::parseLong(&text(offset)) - 1;
    Kokkos::atomic_increment(&rowCounts(r));
    if(line == 0)
      rowCounts(rowCounts.extent(0) - 1) = 0;
  }

  CharView text;
  OffsetView offsets;
  Rowmap rowCounts;
};

template<typename ScalarView, typename RowmapView, typename EntriesView, typename memory_space>
struct CRSFromDenseFunctor
{
  using OffsetView = Kokkos::View<size_t*, memory_space>;
  using CharView = Kokkos::View<char*, memory_space>;
  using Scalar = typename ScalarView::non_const_value_type;
  using Offset = typename RowmapView::non_const_value_type;
  using Ordinal = typename EntriesView::non_const_value_type;

  CRSFromDenseFunctor(Ordinal numRows_, Ordinal numCols_, const ScalarView& values_, const RowmapView& rowmap_, const EntriesView& entries_, const OffsetView& lineOffsets_, const CharView& text_)
    : numRows(numRows_), numCols(numCols_), values(values_), rowmap(rowmap_), entries(entries_), lineOffsets(lineOffsets_), text(text_)
  {}

  //i is the byte index into file
  //if finalPass, lline is the line in which byte i appears
  KOKKOS_INLINE_FUNCTION void operator()(size_t line) const
  {
    size_t lineOffset = lineOffsets(line);
    //Line will only contain one scalar
    Scalar val = DeviceParsing::parseScalar<Scalar>(&text(lineOffset));
    //Lines are ordered column-major, so know exactly what entry this is
    Ordinal row = line % numRows;
    Ordinal col = line / numRows;
    if(col == 0)
    {
      rowmap(row) = (Offset) row * numCols;
      if(line == 0)
        rowmap(numRows) = numRows * numCols;
    }
    Offset offset = row * numCols + col;
    entries(offset) = col;
    values(offset) = val;
  }

  Ordinal numRows;
  Ordinal numCols;
  ScalarView values;
  RowmapView rowmap;
  EntriesView entries;
  OffsetView lineOffsets;
  CharView text;
};

template<typename ScalarView, typename RowmapView, typename EntriesView, typename memory_space>
struct InsertEntriesFunctor
{
  using OffsetView = Kokkos::View<size_t*, memory_space>;
  using CharView = Kokkos::View<char*, memory_space>;
  using IntView = Kokkos::View<int*, memory_space>;
  using Scalar = typename ScalarView::non_const_value_type;
  using Offset = typename RowmapView::non_const_value_type;
  using Ordinal = typename EntriesView::non_const_value_type;

  InsertEntriesFunctor(const CharView& text_, const OffsetView& lineOffsets_, const ScalarView& values_, const RowmapView& rowmap_, const EntriesView& colinds_, bool isPattern_)
    : text(text_), lineOffsets(lineOffsets_), numInserted("Num inserted", rowmap_.extent(0) - 1), values(values_), rowmap(rowmap_), colinds(colinds_), isPattern(isPattern_)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(size_t line) const
  {
    size_t offset = lineOffsets(line);
    char* str = &text(offset);
    Ordinal row = DeviceParsing::parseLong(str) - 1;
    while(*str == ' ' || *str == '\t')
      str++;
    Ordinal col = DeviceParsing::parseLong(str) - 1;
    while(*str == ' ' || *str == '\t')
      str++;
    Scalar val = isPattern ? Kokkos::ArithTraits<Scalar>::one() : DeviceParsing::parseScalar<Scalar>(str);
    Offset pos = rowmap(row) + Kokkos::atomic_fetch_add(&numInserted(row), 1);
    values(pos) = val;
    colinds(pos) = col;
  }

  CharView text;
  OffsetView lineOffsets;
  IntView numInserted;
  ScalarView values;
  RowmapView rowmap;
  EntriesView colinds;
  bool isPattern;
};

namespace FastMMRead
{
  enum MtxObject
  {
    UNDEFINED_OBJECT,
    MATRIX,
    VECTOR
  };
  enum MtxFormat
  {
    UNDEFINED_FORMAT,
    COORDINATE,
    ARRAY
  };
  enum MtxField
  {
    UNDEFINED_FIELD,
    REAL,     //includes both float and double
    COMPLEX,  //includes complex<float> and complex<double>
    INTEGER,  //includes all integer types
    PATTERN   //not a type, but means the value for every entry is 1
  };
  enum MtxSym
  {
    UNDEFINED_SYMMETRY,
    GENERAL,
    SYMMETRIC,      //A(i, j) = A(j, i)
    SKEW_SYMMETRIC, //A(i, j) = -A(j, i)
    HERMITIAN       //A(i, j) = a + bi; A(j, i) = a - bi
  };
}

template<typename ScalarView, typename RowmapView, typename EntriesView>
struct TransformTransposeFunctor
{
  using Scalar = typename ScalarView::non_const_value_type;
  using Offset = typename RowmapView::non_const_value_type;
  using Ordinal = typename EntriesView::non_const_value_type;

  TransformTransposeFunctor(ScalarView values_, RowmapView rowmap_, EntriesView entries_, FastMMRead::MtxSym mode_)
    : values(values_), rowmap(rowmap_), entries(entries_), mode(mode_)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(Ordinal row) const
  {
    Offset rowBegin = rowmap(row);
    Offset rowEnd = rowmap(row + 1);
    for(Offset j = rowBegin; j < rowEnd; j++)
    {
      Ordinal col = entries(j);
      if(row == col)
      {
        //Zero out the diagonal
        values(j) = Kokkos::ArithTraits<Scalar>::zero();
      }
      else
      {
        //Depending on the mode, also modify the off-diagonals
        if(mode == FastMMRead::SKEW_SYMMETRIC)
          values(j) = -values(j);
        else if(mode == FastMMRead::SKEW_SYMMETRIC)
          values(j) = Kokkos::ArithTraits<Scalar>::conj(values(j));
      }
    }
  }

  ScalarView values;
  RowmapView rowmap;
  EntriesView entries;
  const FastMMRead::MtxSym mode;
};

template<typename TeamMember, typename RowmapView, typename EntriesView, typename ScalarView>
struct SortRowsFunctor
{
  using Scalar = typename ScalarView::non_const_value_type;
  using Offset = typename RowmapView::non_const_value_type;
  using Ordinal = typename EntriesView::non_const_value_type;

  SortRowsFunctor(const RowmapView& rowmap_, const EntriesView& entries_, const ScalarView& values_) :
    rowmap(rowmap_), entries(entries_), values(values_)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(const TeamMember t) const
  {
    Ordinal row = t.league_rank();
    Offset rowStart = rowmap(row);
    Offset rowEnd = rowmap(row + 1);
    Offset rowNum = rowEnd - rowStart;
    KokkosKernels::Impl::TeamBitonicSort2<Ordinal, Ordinal, Scalar, TeamMember>
      (entries.data() + rowStart, values.data() + rowStart, rowNum, t);
  }

  RowmapView rowmap;
  EntriesView entries;
  ScalarView values;
};

template <typename crsMat_t>
int read_mtx_device(const char *fileName)
{
  using namespace FastMMRead; //for MatrixMarket format enums
  typedef typename crsMat_t::StaticCrsGraphType graph_t;
  typedef typename graph_t::row_map_type::non_const_type row_map_t;
  typedef typename graph_t::entries_type::non_const_type cols_t;
  typedef typename crsMat_t::values_type::non_const_type values_t;
  typedef typename row_map_t::value_type size_type;
  typedef typename cols_t::value_type   lno_t;
  typedef typename values_t::value_type scalar_t;
  typedef typename crsMat_t::execution_space execution_space;
  typedef typename crsMat_t::memory_space memory_space;
  typedef Kokkos::View<char*, Kokkos::HostSpace> HostCharView;
  typedef Kokkos::View<char*, memory_space> CharView;
  typedef Kokkos::View<size_t*, memory_space> OffsetView;
  FILE* f = fopen(fileName, "r");
  if(!f)
    throw std::runtime_error ("File cannot be opened\n");
  fseek(f, 0, SEEK_END);
  size_t fileLen = ftell(f);
  rewind(f);
  HostCharView fileHost(Kokkos::ViewAllocateWithoutInitializing("MM file contents host"), fileLen);
  fread((void*) fileHost.data(), 1, fileLen, f);
  fclose(f);
  //Now, parse the MatrixMarket header
  size_t iter = 0;
  //scan one line into fline
  std::string fline;
  //note: for CRLF line endings, the '\r' will appear as
  //the last char of fline and won't break anything
  while(iter < fileLen && fileHost(iter) != '\n')
  {
    fline += fileHost(iter);
    iter++;
  }
  if (fline.size() < 2 || fline[0] != '%' || fline[1] != '%'){
    throw std::runtime_error ("Invalid MM file. Line-1\n");
  }
  iter++;
  //this line contains #rows, #cols, #entries
  //make sure every required field is in the file, by initializing them to UNDEFINED_*
  MtxObject mtx_object = UNDEFINED_OBJECT;
  MtxFormat mtx_format = UNDEFINED_FORMAT;
  MtxField mtx_field = UNDEFINED_FIELD;
  MtxSym mtx_sym = UNDEFINED_SYMMETRY;
  if (fline.find("matrix") != std::string::npos){
    mtx_object = MATRIX;
  } else if (fline.find("vector") != std::string::npos){
    mtx_object = VECTOR;
    throw std::runtime_error("MatrixMarket \"vector\" is not supported by KokkosKernels read_mtx()");
  }
  if (fline.find("coordinate") != std::string::npos){
    //sparse
    mtx_format = COORDINATE;
  }
  else if (fline.find("array") != std::string::npos){
    //dense
    mtx_format = ARRAY;
  }
  if(fline.find("real") != std::string::npos || 
     fline.find("double") != std::string::npos)
  {
    if(!std::is_floating_point<scalar_t>::value)
      throw std::runtime_error("scalar_t in read_mtx() incompatible with float or double typed MatrixMarket file.");
    else
      mtx_field = REAL;
  }
  else if (fline.find("complex") != std::string::npos){
    if(!(std::is_same<scalar_t, Kokkos::complex<float>>::value ||
          std::is_same<scalar_t, Kokkos::complex<double>>::value))
      throw std::runtime_error("scalar_t in read_mtx() incompatible with complex-typed MatrixMarket file.");
    else
      mtx_field = COMPLEX;
  }
  else if (fline.find("integer") != std::string::npos){
    if(std::is_integral<scalar_t>::value)
      mtx_field = INTEGER;
    else
      throw std::runtime_error("scalar_t in read_mtx() incompatible with integer-typed MatrixMarket file.");
  }
  else if (fline.find("pattern") != std::string::npos){
    mtx_field = PATTERN;
    //any reasonable choice for scalar_t can represent "1" or "1.0 + 0i", so nothing to check here
  }
  if (fline.find("general") != std::string::npos){
    mtx_sym = GENERAL;
  }
  else if (fline.find("skew-symmetric") != std::string::npos){
    mtx_sym = SKEW_SYMMETRIC;
  }
  else if (fline.find("symmetric") != std::string::npos){
    //checking for "symmetric" after "skew-symmetric" because it's a substring
    mtx_sym = SYMMETRIC;
  }
  else if (fline.find("hermitian") != std::string::npos ||
      fline.find("Hermitian") != std::string::npos){
    mtx_sym = HERMITIAN;
  }
  //Validate the matrix attributes
  if(mtx_format == ARRAY)
  {
    if(mtx_sym == UNDEFINED_SYMMETRY)
      mtx_sym = GENERAL;
    if(mtx_sym != GENERAL)
      throw std::runtime_error("array format MatrixMarket file must have general symmetry (optional to include \"general\")");
  }
  if(mtx_object == UNDEFINED_OBJECT)
    throw std::runtime_error("MatrixMarket file header is missing the object type.");
  if(mtx_format == UNDEFINED_FORMAT)
    throw std::runtime_error("MatrixMarket file header is missing the format.");
  if(mtx_field == UNDEFINED_FIELD)
    throw std::runtime_error("MatrixMarket file header is missing the field type.");
  if(mtx_sym == UNDEFINED_SYMMETRY)
    throw std::runtime_error("MatrixMarket file header is missing the symmetry type.");
  //Then throw away all lines which begin with '%'
  while(true)
  {
    size_t lineBegin = iter;
    fline = "";
    while(iter < fileLen && fileHost(iter) != '\n')
    {
      fline += fileHost(iter);
      iter++;
    }
    iter++;
    if(fline[0] != '%')
      break;
  }
  std::stringstream ss (fline);
  lno_t nr = 0, nc = 0;
  size_type nnz = 0;
  ss >> nr >> nc;
  if(mtx_format == COORDINATE)
    ss >> nnz;
  else
    nnz = nr * nc;
  bool symmetrize = mtx_sym != GENERAL;
  if(symmetrize && nr != nc)
  {
    throw std::runtime_error("A non-square matrix cannot be symmetrized (invalid MatrixMarket file)");
  }
  if(mtx_format == ARRAY)
  {
    //Array format only supports general symmetry and non-pattern
    if(symmetrize)
      throw std::runtime_error("array format MatrixMarket file cannot be symmetrized.");
    if(mtx_field == PATTERN)
      throw std::runtime_error("array format MatrixMarket file can't have \"pattern\" field type.");
  }
  //The remainder of the file (from iter) is a list of entries, separated by newlines.
  //Copy this to device.
  size_t bodyLen = fileLen - iter;
  CharView fileDev(Kokkos::ViewAllocateWithoutInitializing("MM file contents"), bodyLen);
  Kokkos::deep_copy(fileDev, Kokkos::subview(fileHost, std::make_pair(iter, fileLen)));
  //Use a scan to find the offset of each line (position of '\n' + 1)
  OffsetView lineOffsets("MM line offsets", nnz);
  Kokkos::parallel_scan(Kokkos::RangePolicy<execution_space>(0, bodyLen),
      LineOffsetsFunctor<memory_space>(fileDev, lineOffsets));
  row_map_t rowmap(std::string("Rowmap: ") + fileName, nr + 1);
  cols_t entries(Kokkos::ViewAllocateWithoutInitializing(std::string("Colinds: ") + fileName, nnz));
  values_t values(Kokkos::ViewAllocateWithoutInitializing(std::string("Values: ") + fileName, nnz));
  if(mtx_format == ARRAY)
  {
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nnz),
        CRSFromDenseFunctor<values_t, row_map_t, cols_t, memory_space>(nr, nc, values, rowmap, entries, lineOffsets, fileDev));
  }
  else
  {
    //Count the entries in each row
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nnz),
        CountRowEntriesFunctor<row_map_t, memory_space>(fileDev, lineOffsets, rowmap));
    //Prefix-sum to get rowmap
    exclusive_parallel_prefix_sum<row_map_t, execution_space>(nr + 1, rowmap);
    //Insert entries
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nnz),
        InsertEntriesFunctor<values_t, row_map_t, cols_t, memory_space>
        (fileDev, lineOffsets, values, rowmap, entries, mtx_format == PATTERN));
  }
  //At this point, have the full CRS from the file, so free the text and line offset views
  fileHost = HostCharView();
  fileDev = HostCharView();
  lineOffsets = OffsetView();
  //If needed, symmetrize the matrix.
  if(symmetrize)
  {
    row_map_t t_rowmap("Rowmap^T", nr + 1);
    cols_t    t_entries(Kokkos::ViewAllocateWithoutInitializing("Colinds^T"), nnz);
    values_t  t_values(Kokkos::ViewAllocateWithoutInitializing("Values^T"), nnz);
    transpose_matrix<row_map_t, cols_t, values_t, row_map_t, cols_t, values_t, row_map_t, execution_space>
      (nr, nr, rowmap, entries, values, t_rowmap, t_entries, t_values);
    //In all modes, zero out the transpose's diagonal. Also:
    //If mtx_sym == SKEW_SYMMETRIC, multiply off-diagonals by -1.
    //If mtx_sym == HERMITIAN, convert off-diagonals to their conjugates.
    Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, nr),
        TransformTransposeFunctor<values_t, row_map_t, cols_t>
        (values, rowmap, entries, mtx_sym));
    //Then use spadd to combine the orignal and transpose to get the final result
    //Note: spadd always produces a sorted result
    using HandleType = KokkosKernelsHandle<size_type, lno_t, scalar_t, execution_space, memory_space, memory_space>;
    HandleType kkh;
    kkh.create_spadd_handle(false);
    row_map_t a_rowmap = rowmap;
    cols_t a_entries = entries;
    values_t a_values = values;
    rowmap = row_map_t(std::string("Rowmap: ") + fileName, nr + 1);
    KokkosSparse::Experimental::spadd_symbolic<HandleType, row_map_t, cols_t, row_map_t, cols_t, row_map_t, cols_t>
      (&kkh, a_rowmap, a_entries, t_rowmap, t_entries, rowmap);
    size_type finalNNZ = kkh.get_spadd_handle()->get_c_nnz();
    values = values_t(std::string("Values: ") + fileName, finalNNZ);
    entries = cols_t(std::string("Colinds: ") + fileName, finalNNZ);
    KokkosSparse::Experimental::spadd_numeric<HandleType, row_map_t, cols_t, scalar_t, values_t, row_map_t, cols_t, scalar_t, values_t, row_map_t, cols_t, values_t>
      (&kkh, a_rowmap, a_entries, a_values, t_rowmap, t_entries, t_values, rowmap, entries, values);
    kkh.destroy_spadd_handle();
  }
  else if(mtx_format != ARRAY)
  {
    //Didn't run spadd, so need to sort the columns in each row
    using TeamPol = Kokkos::TeamPolicy<execution_space>;
    using TeamMember = typename TeamPol::member_type;
    SortRowsFunctor<TeamMember, row_map_t, cols_t, valuse_t> sortFunctor(rowmap, entries, values);
    //Try to get a team size that's 0.5 times average degree
    int teamSize = (nnz + nr - 1) / nr;
    TeamPol temp(1, teamSize);
    teamSize = std::min(teamSize, temp.team_size_max(sortFunctor, ParallelForTag()));
    Kokkos::parallel_for(TeamPol(nr, teamSize), sortFunctor);
  }
}

}}


#endif
