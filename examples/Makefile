# Makefile example (icc)

SHELL := /bin/bash
CXX=g++-11
CXXFLAGS?= -fopenmp -std=c++20 -O3 -DNDEBUG -I../src -DAMBIENT_MPI=MPI_DISABLE

.DEFAULT: all
.PHONY: all clean

#EXAMPLES = diffusion partitioned_vector vector atomic matrix mp_gemm mp_gemm_simple mp_gemm_log groups future
EXAMPLES = vector
all: $(EXAMPLES)

%: %.cpp
	$(CXX) -o $(basename $<) $< $(CXXFLAGS)

clean:
	-rm -f $(EXAMPLES)
