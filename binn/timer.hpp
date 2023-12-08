#pragma once
#include <chrono>
#include <fmt/format.h>

// a timer class to report the elasped time in milliseconds
namespace binn
{

class Timer
{
  public:
  Timer() : beg_(clock_::now()) {}
  void reset()
  {
    beg_ = clock_::now();
  }

  double elapsed() const
  {
    return std::chrono::duration_cast<time_unit_>(clock_::now() - beg_).count();
  }

  void report(const std::string_view msg = "Elapsed time")
  {
    fmt::print("{}: {} ms\n", msg, elapsed());
  }

  private:
  using clock_     = std::chrono::high_resolution_clock;
  using time_unit_ = std::chrono::milliseconds;
  std::chrono::time_point<clock_> beg_;
};

}    // namespace binn