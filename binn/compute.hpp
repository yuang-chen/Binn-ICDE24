#pragma once
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <omp.h>

#include <atomic>
#include <cassert>
#include <climits>
#include <numeric>
#include <span>
#include <vector>

#include "timer.hpp"
#include "traits.hpp"
#include "variables.hpp"

namespace binn {
template <typename index_t, typename value_t>
void scatter(std::vector<value_t> &x, std::vector<value_t> &buf,
             std::vector<index_t> &itr) {
    index_t index = 0;
    for (const auto ver : itr)  // auto const src_ver: inter_bin)
        buf[index++] = x[ver];
}
template <typename index_t, typename value_t>
void gather(std::vector<value_t> &y, std::vector<value_t> &buf,
            std::vector<value_t> &val, std::vector<index_t> &dst) {
    index_t index = umask<index_t>::max;
    index_t count = 0;

    for (const auto curr_ver : dst) {
        index += (curr_ver >> umask<index_t>::digits);
        const auto verx = curr_ver & umask<index_t>::pos;
        const auto data = buf[index] * val[count++];
        y[verx] += data;
    }
}

template <MatrixType Blocks, typename index_t = Blocks::index_type,
          typename value_t = Blocks::value_type>
auto binn_spmv(Blocks &blocks, std::vector<value_t> &x,
               std::vector<value_t> &y) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(Param::threads)
    for (size_t i = 0; i < blocks.numblk; i++) {
        for (size_t j = 0; j < blocks.numblk; j++) {
            scatter<index_t, value_t>(x, blocks.bufbin[i, j],
                                      blocks.colbin[i, j]);
        }
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(Param::threads)
    for (size_t i = 0; i < blocks.numblk; i++) {
        for (size_t j = 0; j < blocks.numblk; j++) {
            gather<index_t, value_t>(y, blocks.bufbin[j, i],
                                     blocks.valbin[j, i], blocks.rowbin[j, i]);
        }
    }
}

}  // namespace binn