#include "ambient/ambient.hpp"
#include "ambient/container/numeric/matrix.hpp"
#include "utils/timer.hpp"

#define N AMBIENT_DEFAULT_IB*8

struct Decomposition {
    void init(int np, int nq){
        np_ = np; nq_ = nq;
        std::cout << "Proc grid: " << np << "x" << nq << "\n";
    }
    ambient::scope::const_iterator operator()(int i, int j){
        return (ambient::scope::begin() + (i % np_) * nq_ + j % nq_);
    }
    int np_;
    int nq_;
} where;

int main(int argc, char* argv[]){
    using matrix = ambient::matrix<double>;
    if(argc > 2) where.init(std::stoi(argv[1]), std::stoi(argv[2]));
    else where.init(ambient::scope::size(), 1);
    ambient::tiles< matrix > a(N,N), b(N,N), c(N,N);

    for(int i = 0; i < a.mt; i++)
    for(int j = 0; j < a.nt; j++){
        ambient::actor proc(where(i,j));
        ambient::numeric::fill_random(a.tile(i,j));
        ambient::numeric::fill_random(b.tile(i,j));
    }

    ambient::timer t1("gemm"); t1.begin(); std::cout << "gemm (" << N << ")\n";

    for(int k = 0; k < a.nt; k++)
    for(int j = 0; j < c.nt; j++)
    for(int i = 0; i < c.mt; i++){
        ambient::actor proc(where(i,j));
        ambient::numeric::gemm_fma(a.tile(i,k), b.tile(k,j), c.tile(i,j));
    }

    t1.end();
    return 0;
}

