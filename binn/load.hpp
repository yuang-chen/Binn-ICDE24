#pragma once

#include "structures.hpp"
#include "traits.hpp"
#include <cstdlib>
#include <fmt/core.h>
#include <fstream>
#include <regex>
#include <string>

namespace binn
{

template <MatrixType Matrix>
void sort_by_column(Matrix &matrix)
{
  using index_t = index_t_rm_cvr<decltype(matrix)>;

  const auto  nnz    = matrix.nnz;
  const auto  nrow   = matrix.nrow;
  const auto &colidx = matrix.colidx;

  std::vector<index_t> indices(nnz);
  std::iota(indices.begin(), indices.end(), 0);

  std::stable_sort(indices.begin(), indices.end(), [&colidx](index_t a, index_t b) {
    return colidx[a] < colidx[b];
  });

  std::vector<index_t> rowsrt(nnz);
  std::vector<index_t> colsrt(nnz);
  std::vector<index_t> colptr(nrow + 1);

  for (size_t i = 0; i < nnz; ++i)
  {
    rowsrt[i] = matrix.rowidx[indices[i]];
    colsrt[i] = matrix.colidx[indices[i]];
  }

  for (int i = 0; i < nnz; ++i)
  {
    colptr[colidx[i] + 1]++;
  }
  std::partial_sum(colptr.begin(), colptr.end(), colptr.begin());

  // #pragma omp parallel for
  //   for (index_t i = 0; i < nrow; i++)
  //   {
  //     for (index_t j = colptr[i]; j < colptr[i + 1]; j++)
  //     {
  //       std::sort(rowsrt.begin() + colptr[i], rowsrt.begin() + colptr[i + 1]);
  //     }
  //   }

  std::vector<index_t> tmp1;
  std::vector<index_t> tmp2;
  for (size_t i = 0; i < nnz; ++i)
  {
    if (colsrt[i] == 1)
    {
      tmp1.push_back(rowsrt[i]);
      tmp2.push_back(i);
    }
  }
  fmt::print("tmp1: {}\n", tmp1);
  fmt::print("tmp2: {}\n", tmp2);
  matrix.rowidx = std::move(rowsrt);
  matrix.colidx = std::move(colsrt);
  matrix.cscptr = std::move(colptr);
}

template <MatrixType Matrix>
void load_binary_edgelist(Matrix &matrix, std::string_view filename)
{
  using index_t = index_t_rm_cvr<decltype(matrix)>;

  std::ifstream input_file(std::string(filename), std::ios::binary);
  if (!input_file.is_open())
  {
    fmt::print("cannot open the input mix file!\n");
    std::exit(1);
  }
};

template <MatrixType Matrix>
void load_edgelist(Matrix &matrix, std::string_view filename)
{
  using index_t = index_t_rm_cvr<decltype(matrix)>;

  std::ifstream input_file(std::string(filename), std::ios::binary);
  if (!input_file.is_open())
  {
    fmt::print("cannot open the input csr file!\n");
    std::exit(1);
  }
};

template <MatrixType Matrix>
void load_csr_convert_coo(Matrix &matrix, std::string_view filename)
{
  using index_t = index_t_rm_cvr<decltype(matrix)>;

  std::ifstream input_file(std::string(filename), std::ios::binary);
  if (!input_file.is_open())
  {
    fmt::print("cannot open the input csr file!\n");
    std::exit(1);
  }

  index_t nrow = 0;
  index_t nnz  = 0;
  input_file.read(reinterpret_cast<char *>(&nrow), sizeof(index_t));
  input_file.read(reinterpret_cast<char *>(&nnz), sizeof(index_t));

  std::vector<index_t> csrPtr(nrow + 1);    // +2 for padding
  std::vector<index_t> csrIdx(nnz);
  input_file.read(reinterpret_cast<char *>(csrPtr.data()), nrow * sizeof(index_t));
  input_file.read(reinterpret_cast<char *>(csrIdx.data()), nnz * sizeof(index_t));

  csrPtr[nrow] = nnz;
  input_file.close();

  matrix.rowidx.resize(nnz);
#pragma omp parallel for
  for (index_t i = 0; i < nrow; i++)
  {
    for (index_t j = csrPtr[i]; j < csrPtr[i + 1]; j++)
    {
      matrix.rowidx[j] = i;
      std::sort(csrIdx.begin() + csrPtr[i], csrIdx.begin() + csrPtr[i + 1]);
    }
  }
  matrix.nrow   = nrow;
  matrix.nnz    = nnz;
  matrix.colidx = std::move(csrIdx);
  matrix.csrptr = std::move(csrPtr);
};

template <MatrixType Matrix>
void load_mix(Matrix &matrix, std::string_view filename)
{
  using index_t = index_t_rm_cvr<decltype(matrix)>;

  std::ifstream input_file(std::string(filename), std::ios::binary);
  if (!input_file.is_open())
  {
    fmt::print("cannot open the input mix file!\n");
    std::exit(1);
  }

  input_file.read(reinterpret_cast<char *>(&matrix.nrow), sizeof(index_t));
  input_file.read(reinterpret_cast<char *>(&matrix.nnz), sizeof(index_t));

  matrix.rowidx.resize(matrix.nnz);
  matrix.colidx.resize(matrix.nnz);
  matrix.csrptr.resize(matrix.nrow + 1);
  matrix.cscptr.resize(matrix.nrow + 1);
  matrix.csridx.resize(matrix.nnz);
  matrix.cscidx.resize(matrix.nnz);

  input_file.read(reinterpret_cast<char *>(matrix.csrptr.data()), matrix.nrow * sizeof(index_t));
  input_file.read(reinterpret_cast<char *>(matrix.csridx.data()), matrix.nnz * sizeof(index_t));
  input_file.read(reinterpret_cast<char *>(matrix.cscptr.data()), matrix.nrow * sizeof(index_t));
  input_file.read(reinterpret_cast<char *>(matrix.cscidx.data()), matrix.nnz * sizeof(index_t));
  input_file.close();

  matrix.csrptr[matrix.nrow] = matrix.nnz;
  matrix.cscptr[matrix.nrow] = matrix.nnz;

#pragma omp parallel for
  for (index_t i = 0; i < matrix.nrow; i++)
  {
    for (index_t j = matrix.cscptr[i]; j < matrix.cscptr[i + 1]; j++)
    {
      matrix.colidx[j] = i;
    }
  }
  // matrix.rowidx = std::move(matrix.cscidx);
  std::copy(matrix.cscidx.begin(), matrix.cscidx.end(), matrix.rowidx.begin());
}

template <MatrixType Matrix>
void load(Matrix &matrix, std::string_view filename)
{
  fmt::print("--------matrix loading--------\n");

  if (filename.ends_with("csr"))
  {
    fmt::print("loading {}...\n", filename);
    load_csr_convert_coo(matrix, filename);
  }
  else if (filename.ends_with("mix"))
  {
    load_mix(matrix, filename);
  }

  else
  {
    fmt::print("unsupported matrix format\n");
    std::exit(1);
  }
  // sort_by_column(matrix);

  using index_t = index_t_rm_cvr<decltype(matrix)>;
  using value_t = value_t_rm_cvr<decltype(matrix)>;

  fmt::print("#vertices: {}, #edges: {}\n", matrix.nrow, matrix.nnz);
  fmt::print("index_t: {} bytes, value_t: {} bytes\n", sizeof(index_t), sizeof(value_t));

  matrix.values.resize(matrix.nnz);

#pragma omp parallel for
  for (index_t i = 0; i < matrix.nnz; i++)
    matrix.values[i] = 1;    // i % 13;    // mtx pattern

  // if (Param::output) storeMIX(filename);
};

}    // namespace binn
