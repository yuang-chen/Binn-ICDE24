#include <fmt/format.h>
#include <chrono>
#include "compute.hpp"
#include "divide.hpp"
#include "load.hpp"
#include "options.hpp"
#include "variables.hpp"

int main(int argc, char **argv) {
    using I = unsigned;
    using T = double;
    binn::program_options(argc, argv);

    binn::Matrix<I, T> matrix{};
    binn::load(matrix, binn::Param::input_file);

    if (binn::Param::transpose) {
        matrix.transpose();
    }
    binn::Timer timer{};
    binn::Blocks<decltype(matrix)> blocks(matrix.nrow, binn::Param::threads,
                                          binn::Param::submatrix_size);
    binn::partition(matrix, blocks);
    timer.report("preprocessing");

    std::vector<T> x(matrix.nrow);
    std::vector<T> y(matrix.nrow);
    std::iota(x.begin(), x.end(), 0);
    std::fill(y.begin(), y.end(), 0);

    timer.reset();
    double time = 0;
    for (int i = 0; i < binn::Param::rounds; i++) {
        for (int j = 0; j < binn::Param::iterations; j++) {
            std::fill(y.begin(), y.end(), 0);
            binn::binn_spmv(blocks, x, y);
        }
    }
    fmt::print("elapsed time: {} ms\n", timer.elapsed());
    // fmt::print("max time: {} ms\n", time / binn::Param::rounds /
    // binn::Param::iterations);

    // for verification
    fmt::print("First 10 elements: {}\n",
               fmt::join(y.begin(), y.begin() + 10, ", "));
    fmt::print("Total value of y: {}\n",
               std::accumulate(y.begin(), y.end(), 0.0));

    return 0;
}
