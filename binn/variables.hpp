#pragma once
#include <array>
#include <limits>
#include <string>

namespace binn
{

enum class DType
{
  in  = 0,
  out = 1
};

enum class RAlgo
{
  original = 0,
  pop      = 1,
  bihub    = 2,
  trihub   = 3,
  fbc      = 4,
  dbg      = 5,
  rnd      = 6,
  sort     = 7
};

static constexpr std::array<std::string_view, 2> BoolStr{"No", "Yes"};
static constexpr std::array<std::string_view, 2> DTypeName{"InDegree", "OutDegree"};
static constexpr std::array<std::string_view, 8>
    RAlgoName{"Origin", "PoP", "BiHub", "TirHub", "FBC", "DBG", "RND", "Sort"};

template <typename T>
  requires std::is_unsigned<T>::value
struct umask
{
  static constexpr T digits = std::numeric_limits<T>::digits - 1;    //
  static constexpr T msb    = static_cast<T>(1) << digits;
  static constexpr T max    = std::numeric_limits<T>::max();
  static constexpr T pos    = max >> 1;
};

struct Param
{
  static inline unsigned    submatrix_size = 256;    // 512 KB
  static inline unsigned    threads        = 20;
  static inline unsigned    iterations     = 1;
  static inline unsigned    rounds         = 1;
  static inline bool        unequal        = true;
  static inline bool        transpose      = false;
  static inline RAlgo       ralgo          = RAlgo::original;
  static inline DType       dtype          = DType::in;
  static inline size_t      root_vertex    = std::numeric_limits<size_t>::max();
  static inline std::string input_file;
};

}    // namespace binn
