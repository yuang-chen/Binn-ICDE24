#pragma once

#include <concepts>
#include <type_traits>
namespace binn
{

template <typename>
struct matrix_traits;

template <template <typename, typename> class Matrix, typename I, typename T>
struct matrix_traits<Matrix<I, T>>
{
  using index_type = I;
  using value_type = T;
};

template <class M>
using matrix_index_t = typename matrix_traits<M>::index_type;

template <class M>
using matrix_value_t = typename matrix_traits<M>::value_type;

template <class M>
using index_t_rm_cvr = matrix_index_t<std::remove_cvref_t<M>>;

template <class M>
using value_t_rm_cvr = matrix_value_t<std::remove_cvref_t<M>>;

template <class M>
concept MatrixType = requires(M matrix) {
  typename matrix_index_t<M>;
  requires std::unsigned_integral<matrix_index_t<M>>;    // the index_type should be unsigned
  typename matrix_value_t<M>;
  requires std::is_same_v<matrix_value_t<M>, float> || std::is_same_v<matrix_value_t<M>, double>
               || std::is_same_v<matrix_value_t<M>, int>
               || std::is_same_v<matrix_value_t<M>, unsigned int>;
};

}    // namespace binn