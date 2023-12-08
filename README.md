# Binn: Accelerating SpMV for Scale-Free Graphs with Optimized Bins

## Requirement
```shell
openmp
libfmt
```

## How to build
```shell
mkdir build
cd build
cmake ..
make -j
```

## How to run
```shell
./build/app/binning_spmv -f mini-data/wiki.csr
```
The graph must be formatted in binary CSR format, or a mix of CSR + CSC. 