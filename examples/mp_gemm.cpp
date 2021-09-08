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

template<typename T>
inline T& reduce_sum(std::vector<T>& seq, std::pair<int,int> root){
    int n = seq.size();
    for(int s = 1; s < n; s *= 2)
    for(int k = s; k < n; k += s*2){
        ambient::actor proc(where( root.first, (root.second +k-s) % n ));
        seq[k-s] += seq[k];
    }
    return seq[0];
}

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

    for(int ii = 0; ii < a.mt / where.np_; ii++)
    for(int kk = 0; kk < b.nt / where.nq_; kk++){
        for(int i = ii*where.np_; i < (ii+1)*where.np_; i++){ // a.mt
            for(int k = kk*where.nq_; k < (kk+1)*where.nq_; k++){ // b.nt
                std::vector< matrix > ctree(a.nt, matrix(AMBIENT_DEFAULT_IB,AMBIENT_DEFAULT_IB));
                for(int j = 0; j < a.nt; j++){
                    ambient::actor proc(where(i,j));
                    ambient::numeric::gemm(a.tile(i,j), b.tile(j,k), ctree[   (a.nt-k + j) % a.nt   ]);
                }
                c.tile(i,k) = reduce_sum(ctree, std::make_pair(i,k));
            }
        }
        ambient::sync();
    }

    t1.end();
    return 0;
}

