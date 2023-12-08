#pragma once

#include <atomic>
#include <cstddef>
#include <numeric>
#include <vector>
#include "traits.hpp"

namespace binn {

template <typename T> class Array2D {
public:
    Array2D() : rows_(0), cols_(0) {}

    Array2D(size_t rows, size_t cols) : rows_(rows), cols_(cols) { allocate(); }

    using Iterator = typename std::vector<T>::iterator;

    Iterator begin() { return data_.begin(); }

    Iterator end() { return data_.end(); }

    T &operator[](size_t r, size_t c) { return data_[r * cols_ + c]; }

    const T &operator[](size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    size_t rows() const { return rows_; }

    size_t cols() const { return cols_; }

    void allocate(size_t row, size_t col) {
        rows_ = row;
        cols_ = col;
        allocate();
    }

private:
    void allocate() { data_.resize(rows_ * cols_); }
    size_t rows_;
    size_t cols_;

    std::vector<T> data_;
};

template <typename T> class Vector3D {
public:
    Vector3D() : rows_(0), cols_(0), volume_(0) {}

    Vector3D(size_t rows, size_t cols, size_t volume)
        : rows_(rows), cols_(cols), volume_(volume) {
        allocate();
    }

    void reserve(size_t rows, size_t cols, size_t volume) {
        rows_ = rows;
        cols_ = cols;
        volume_ = volume;
        reserve();
    }

    T &operator[](size_t r, size_t c, size_t d) { return data_[r][c][d]; }

    const T &operator[](size_t r, size_t c, size_t d) const {
        return data_[r][c][d];
    }

    size_t rows() const { return rows_; }

    size_t cols() const { return cols_; }

    size_t volume() const { return volume_; }

    std::vector<T> &operator[](size_t r, size_t c) { return data_[r][c]; }

    const std::vector<T> &operator[](size_t r, size_t c) const {
        return data_[r][c];
    }

    std::vector<std::vector<T>> &operator[](size_t r) { return data_[r]; }

    const std::vector<std::vector<T>> &operator[](size_t r) const {
        return data_[r];
    }

    inline void allocate() {
        data_.resize(rows_);
        for (size_t r = 0; r < rows_; ++r) {
            data_[r].resize(cols_);
            for (size_t c = 0; c < cols_; ++c) {
                data_[r][c].resize(volume_);
            }
        }
    }

    inline void allocate(size_t rows, size_t cols, auto &volume) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows_);
        //#pragma omp parallel for
        for (size_t r = 0; r < rows_; ++r) {
            data_[r].resize(cols_);
            for (size_t c = 0; c < cols_; ++c) {
                data_[r][c].resize(volume[r, c]);
            }
        }
    }

    inline void reserve() {
        data_.resize(rows_);
        for (size_t r = 0; r < rows_; ++r) {
            data_[r].resize(cols_);
            for (size_t c = 0; c < cols_; ++c) {
                data_[r][c].reserve(volume_);
            }
        }
    }

    size_t rows_;
    size_t cols_;
    size_t volume_;
    // std::vector<T> **data_;
    std::vector<std::vector<std::vector<T>>> data_;
};

template <typename T> class Vector2D {
public:
    Vector2D() : rows_(0), cols_(0) {}

    Vector2D(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        allocate();
    }

    T &operator[](size_t r, size_t c) { return data_[r][c]; }

    const T &operator[](size_t r, size_t c) const { return data_[r][c]; }

    size_t rows() const { return rows_; }

    size_t cols() const { return cols_; }

    const std::vector<T> &operator[](size_t r) const { return data_[r]; }

    std::vector<T> &operator[](size_t r) { return data_[r]; }

    void allocate(size_t row, size_t col) {
        rows_ = row;
        cols_ = col;
        allocate();
    }

    void resize(size_t row) {
        rows_ = row;
        data_.resize(rows_);
    }

    void resize(size_t row, size_t col) {
        cols_ = col;
        data_[row].resize(cols_);
    }

private:
    void allocate() {
        data_.resize(rows_);
        for (size_t r = 0; r < rows_; ++r) {
            data_[r].resize(cols_);
        }
    }
    size_t rows_;
    size_t cols_;

    std::vector<std::vector<T>> data_;
};

//< define a Blocks structure
template <typename index_t, typename value_t> struct BlocksType {
    using index_type = index_t;
    using value_type = value_t;

    index_t base_volume{0};  // #vertices in a block
    index_t base_offset{0};  // log2(base_volume)

    index_t numblk{0};  // numblk the number of  row-wise blocks

    std::vector<index_t> rowrng;  // [numblk*2] 2n: start, 2n+1: stop
    std::vector<index_t> valrng;  // [numblk*2] 2n: start, 2n+1: stop
    std::vector<index_t> div;     //
    std::vector<index_t> dyn_offset;
    std::vector<index_t> dyn_map;

    Vector3D<index_t> ptrbin;
    Vector3D<index_t> colbin;
    Vector3D<index_t> rowbin;
    Vector3D<value_t> bufbin;
    Vector3D<value_t> valbin;

    Vector2D<index_t> colidx;
    Vector2D<index_t> rowidx;

    Vector2D<index_t> colbinn;
    Vector2D<index_t> rowbinn;
    Vector2D<value_t> bufbinn;
    Vector2D<value_t> valbinn;

    BlocksType(const index_t nrow, const unsigned threads,
               const unsigned submatrix_size) {
        base_volume =
            submatrix_size * 1024 / sizeof(index_t);  // submatrix_size is in KB

        numblk = myceil(nrow, base_volume);
        blocks_over_threads(nrow, threads, numblk);
        base_offset = (index_t)log2(base_volume);
    }

    void inline blocks_over_threads(const index_t size, const unsigned threads,
                                    index_t &numblk) {
        while (numblk < 4)  // threads * 4)
        {
            base_volume /= 2;
            numblk =
                myceil(size, base_volume);  // (elements - 1) / base_volume + 1;
        }
    }

    inline index_t myceil(const index_t data, const index_t step) {
        return ((data - 1) / step + 1);
    }

    index_t inline locateBlk(index_t vertex) {
        //! if unequal caching
        const auto anchor = vertex >> base_offset;
        const auto anchored_blk = dyn_offset[anchor];
        const auto chain = (((1 << (base_offset - anchored_blk))) - 1) &
                           (vertex >> anchored_blk);
        return dyn_map[anchor] + chain;
        //! if equal caching
        // return vertex >> offset;
    }
};

//< template alias for BlocksType;
template <MatrixType Mat>
using Blocks = BlocksType<typename Mat::index_t, typename Mat::value_t>;

template <typename I = unsigned, typename T = float> class Matrix {
public:
    using index_t = I;
    using value_t = T;
    ///////////////////
    // hot, cold vertices
    //////////////////
    index_t nrow{0};
    index_t nnz{0};

    std::vector<index_t> rowidx;
    std::vector<index_t> colidx;
    std::vector<index_t> csrptr;  // edge index
    std::vector<index_t> csridx;  // edge index
    std::vector<index_t> cscptr;  // edge index
    std::vector<index_t> cscidx;  // edge index

    std::vector<value_t> values;

    Matrix() = default;
    Matrix(index_t _nrow, index_t _nnz) : nrow(_nrow), nnz(_nnz){};

    //< template for COO Matrix;
    auto transpose() {
        std::swap(csrptr, cscptr);
        std::swap(csridx, cscidx);
        std::swap(rowidx, colidx);
    }
};

}  // namespace binn
