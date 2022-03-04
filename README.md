# MPI Master/Workers Minimal Example

Calculates the dot product of two random vectors by using MPI (very synthetic example, much slower and worse floating-point errors).

## Build Instructions

Ensure Open MPI is installed.

```
git clone https://github.com/kim366/MPI-Master-Workers-Minimal.git
cd MPI-Master-Workers-Minimal
mkdir build
cd build
cmake ..
make
mpiexec -n 4 ./hello
```
