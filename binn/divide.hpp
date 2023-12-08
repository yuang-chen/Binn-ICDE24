#pragma once
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

#include "structures.hpp"
#include "traits.hpp"
#include "variables.hpp"
namespace binn {

template <MatrixType Matrix, MatrixType Blocks>
void unequal_split(const Matrix &matrix, Blocks &blocks, const bool unequal) {
  using index_t = index_t_rm_cvr<decltype(matrix)>;

  fmt::print("--------cache blocking--------\n");
  const index_t avgdeg = matrix.nnz / blocks.numblk;
  const index_t init_numblk = blocks.numblk;  // static numblk
  const index_t base_vol = blocks.base_volume;

  std::vector<index_t> degsum(init_numblk);
  std::vector<index_t> div(init_numblk, 1);

  /////////////////////////////
  // perform the unequal partitioning based on the out-degrees
  // when a submatrix is overflowed, i.e., degsum > 4 * avgdeg,
  // such submatrix is further subdivided until the degsumre <= 4 * avgdeg
  /////////////////////////////
  auto &new_numblk = blocks.numblk;
#pragma omp parallel for reduction(+ : new_numblk)
  for (index_t n = 0; n < init_numblk; n++) {
    auto start = n * base_vol;
    auto end = std::min((n + 1) * base_vol, matrix.nrow);
    degsum[n] = matrix.cscptr.at(end) - matrix.cscptr.at(start);
    if (degsum[n] >= 4 * avgdeg && unequal) {
      // std::log2(degsum[n] / avgdeg)
      div[n] =
          std::pow(2, std::round(std::log2(degsum[n]) / std::log2(avgdeg)));
      new_numblk += div[n] - 1;
    }
  }
  std::vector<index_t> tmp_volume(new_numblk, base_vol);

  blocks.rowrng.resize(new_numblk * 2);
  blocks.valrng.resize(new_numblk * 2);

  blocks.dyn_offset.resize(init_numblk);
  blocks.dyn_map.resize(init_numblk);

#pragma omp parallel for
  for (index_t n = 0; n < init_numblk; n++) {
    const auto n_map = std::accumulate(div.begin(), div.begin() + n, 0);
    blocks.dyn_map[n] = n_map;
    blocks.dyn_offset[n] = blocks.base_offset - log2(div[n]);
    for (index_t i = 0; i < div[n]; i++) {
      const auto idx = i + n_map;
      tmp_volume[idx] = tmp_volume[idx] / div[n];
      blocks.rowrng[idx * 2] = n * base_vol + i * tmp_volume[n_map];
      blocks.rowrng[idx * 2 + 1] =
          std::min(n * base_vol + (i + 1) * tmp_volume[n_map], matrix.nrow);
      blocks.valrng[idx * 2] = matrix.cscptr[blocks.rowrng[idx * 2]];
      blocks.valrng[idx * 2 + 1] = matrix.cscptr[blocks.rowrng[idx * 2 + 1]];
    }
  }
  fmt::print("init_numblk {} new_numblk {}\n", init_numblk, new_numblk);
}

size_t round_down_to_power_of_two(size_t n) {
  size_t power_of_two = 1;
  while ((power_of_two << 1) <= n) {
    power_of_two <<= 1;
  }
  return power_of_two;
}

template <MatrixType Matrix, MatrixType Blocks>
void config_blocks(const Matrix &matrix, Blocks &blocks) {
  using index_t = index_t_rm_cvr<decltype(blocks)>;
  fmt::print("--------binning--------\n");

  const auto numblk = blocks.numblk;
  const auto volume = blocks.base_volume;

  blocks.colbin.reserve(numblk, numblk, volume / 2);
  blocks.rowbin.reserve(numblk, numblk, volume / 2);
  blocks.bufbin.reserve(numblk, numblk, volume / 2);
  blocks.valbin.reserve(numblk, numblk, volume / 2);

  //>> COO partition. COO is sorted by columns
#pragma omp parallel for schedule(dynamic, 1) num_threads(Param::threads)
  for (index_t n = 0; n < numblk; n++) {
    index_t thisblk = numblk;
    index_t thiscol = matrix.nrow;
    for (auto i = blocks.valrng[2 * n]; i < blocks.valrng[2 * n + 1]; i++) {
      if (thiscol != matrix.colidx[i]) {
        thisblk = numblk;  // update thisblk whenever thiscol changes
        thiscol = matrix.colidx[i];
      }
      auto row = matrix.rowidx[i];
      auto blkrow = blocks.locateBlk(row);
      auto blkcol = blocks.locateBlk(thiscol);

      if (thisblk != blkrow) {
        blocks.colbin[blkcol, blkrow].push_back(thiscol);
        row |= umask<index_t>::msb;
        thisblk = blkrow;
      }
      blocks.rowbin[blkcol, blkrow].push_back(row);
      blocks.valbin[blkcol, blkrow].push_back(matrix.values[i]);
    }
  }

#pragma omp parallel for schedule(dynamic, 1) num_threads(Param::threads)
  for (index_t i = 0; i < numblk; i++) {
    for (index_t j = 0; j < numblk; j++) {
      blocks.bufbin[i, j].resize(blocks.colbin[i, j].size());
      blocks.rowbin[i, j].shrink_to_fit();
      blocks.colbin[i, j].shrink_to_fit();
      blocks.valbin[i, j].shrink_to_fit();
    }
  }
}

template <MatrixType Matrix, MatrixType Blocks>
void config_blocks_csc(const Matrix &matrix, Blocks &blocks) {
  using index_t = index_t_rm_cvr<decltype(blocks)>;
  fmt::print("--------binning--------\n");

  const auto numblk = blocks.numblk;
  const auto volume = blocks.base_volume;

  Array2D<index_t> rowcnt(numblk, numblk);  // notice the offset
  Array2D<index_t> colcnt(numblk, numblk);

#pragma omp parallel for schedule(dynamic, 1) num_threads(Param::threads)
  for (index_t n = 0; n < numblk; n++) {
    // thisblk = 0;
    for (auto i = blocks.rowrng[2 * n]; i < blocks.rowrng[2 * n + 1]; i++) {
      auto thisblk = numblk;
      auto col = i;
      for (auto j = matrix.cscptr[i]; j < matrix.cscptr[i + 1]; j++) {
        auto row = matrix.rowidx[j];
        auto blkrow = blocks.locateBlk(row);

        if (thisblk != blkrow) {
          colcnt[n, blkrow]++;
          thisblk = blkrow;
        }
        rowcnt[n, blkrow]++;
      }
    }
  }

  fmt::print("total colbin size {}\n",
             std::accumulate(colcnt.begin(), colcnt.end(), 0));
  fmt::print("total rowbin size {}\n",
             std::accumulate(rowcnt.begin(), rowcnt.end(), 0));

  blocks.colbin.allocate(numblk, numblk, colcnt);
  blocks.bufbin.allocate(numblk, numblk, colcnt);
  blocks.rowbin.allocate(numblk, numblk, rowcnt);
  blocks.valbin.allocate(numblk, numblk, rowcnt);

  std::fill(colcnt.begin(), colcnt.end(), 0);
  std::fill(rowcnt.begin(), rowcnt.end(), 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(Param::threads)
  for (index_t n = 0; n < numblk; n++) {
    // thisblk = 0;
    for (auto i = blocks.rowrng[2 * n]; i < blocks.rowrng[2 * n + 1]; i++) {
      auto thisblk = numblk;
      auto col = i;
      for (auto j = matrix.cscptr[i]; j < matrix.cscptr[i + 1]; j++) {
        auto row = matrix.rowidx[j];
        auto blkrow = blocks.locateBlk(row);

        if (thisblk != blkrow) {
          blocks.colbin[n, blkrow][colcnt[n, blkrow]++] = col;
          row |= umask<index_t>::msb;
          thisblk = blkrow;
        }
        blocks.rowbin[n, blkrow][rowcnt[n, blkrow]] = row;
        blocks.valbin[n, blkrow][rowcnt[n, blkrow]++] = matrix.values[j];
      }
    }
  }
}

//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//< paritition the matrix into blocks
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

template <MatrixType Matrix, MatrixType Blocks>
auto partition(const Matrix &matrix, Blocks &blocks) {
  unequal_split(matrix, blocks, Param::unequal);
  config_blocks_csc(matrix, blocks);
}

}  // namespace binn