
#pragma once
#include "variables.hpp"
#include <fmt/core.h>
#include <getopt.h>
#include <string>

namespace binn
{

auto program_options(int argc, char *argv[])
{
  int opt;
  if (argc == 1)
  {
    fmt::print("Usage: {} ...\n"
               "              [-f input_file]\n"
               "              [-s submatrix_size]\n"
               "              [-r rounds]\n"
               "              [-t transpose]\n"
               "              [-p transpose]\n"
               "              [-i iterations]\n"
               "              [-v root_vertex]\n"
               "              [-d degree_type (0:in 1:out)]\n"
               "              [-a reorder_algo (0:origin 1:bihub 2:trihub 3:pop 4:fbc 5:dbg 6:rnd "
               "7:sort)]\n"
               "              [-u unequal_split (1|0)]\n",
               argv[0]);
    std::exit(EXIT_FAILURE);
  }
  Param::threads = omp_get_max_threads();
  while ((opt = getopt(argc, argv, "s:f:v:t:p:i:r:d:u:a:")) != -1)
  {
    switch (opt)
    {
    case 's':
      Param::submatrix_size = std::stoi(optarg);
      break;
    case 'f':
      Param::input_file = optarg;
      break;
    case 'v':
      Param::root_vertex = std::stoi(optarg);
      break;
    case 't':
      Param::threads = std::stoi(optarg);
      break;
    case 'p':
      Param::transpose = std::stoi(optarg);
      break;
    case 'i':
      Param::iterations = std::stoi(optarg);
      break;
    case 'r':
      Param::rounds = std::stoi(optarg);
      break;
    case 'a':
      Param::ralgo = static_cast<RAlgo>(std::stoi(optarg));
      break;
    case 'u':
      Param::unequal = std::stoi(optarg);
      break;
    default:
      fmt::print("Usage: {} ...\n"
                 "              [-s submatrix_size]\n"
                 "              [-r rounds]\n"
                 "              [-i iterations]\n"
                 "              [-f input_file]\n"
                 "              [-v root_vertex]\n"
                 "              [-a reorder_algo (0:origin 1:bihub 2:trihub 3:pop 4:fbc 5:dbg "
                 "6:rnd 7:sort)]\n"
                 "              [-u unequal_split (1|0)]\n"
                 "              [-t transpose or not (1|0)]\n",
                 argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  fmt::print("--------experimental setting--------\n");
  fmt::print("threads: {0}, rounds: {1}, iterations: {2}, sub_size: {3}KB,\n"
             "reorder algo: {4}, unequal split: {5}, transpose: {6}\n",
             Param::threads,
             Param::rounds,
             Param::iterations,
             Param::submatrix_size,
             RAlgoName[static_cast<int>(Param::ralgo)],
             BoolStr[static_cast<int>(Param::unequal)],
             BoolStr[static_cast<int>(Param::transpose)]);
}
}    // namespace binn
