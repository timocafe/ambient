/*
 * Copyright Institute for Theoretical Physics, ETH Zurich 2015.
 * Distributed under the Boost Software License, Version 1.0.
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef AMBIENT_UTILS_MKL_PARALLEL
#define AMBIENT_UTILS_MKL_PARALLEL

 /*
  * Most of the time single-threaded MKL calls are sufficient - but in some cases
  * there could be not enough tasks to be executed and therefor one would benefit
  * from MKL's inner parallelisation. And while MKL_NUM_THREADS environment variable
  * controls the number of threads in ALL of the MKL calls - it would lead to resource
  * overloading (and hence performance degradation) when there is no need for task's
  * inner parallelism.
  *
  * mkl_parallel acts as a guard - if used, only the provided number of threads is
  * used inside MKL calls during its lifetime, single thread is used anywhere else.
  * Without an argument mkl_parallel will look for MKL_NUM_THREADS and will result
  * in no op if it's not present.
  *
  * Important: set AMBIENT_MKL_PARALLEL environment variable to enable this utility.
  */

#include <dlfcn.h>

extern "C" {
    void MKL_Set_Num_Threads(int nth);
}

namespace ambient {

    inline int get_default_nt();

    class mkl_parallel {
    public:
        typedef void (*fptr_t)(int);
        mkl_parallel(int n = 0) : nt(n) {
            if (nt <= 0) nt = get_default_nt();
            else if (!get_default_nt()) nt = 0;
            if (nt) invoke_set_num_threads(nt);
        }
        ~mkl_parallel() {
            if (nt) invoke_set_num_threads(1);
        }
    private:
        void invoke_set_num_threads(int nt) {
            static fptr_t fptr = NULL;
            if (fptr == NULL) {
                void* handle = dlopen("libmkl_intel_lp64.so", RTLD_LAZY);
                if (!handle) throw std::runtime_error("Error: cannot open libmkl_intel_lp64.so!");
                dlerror(); // reset errors
                fptr = (fptr_t)dlsym(handle, "MKL_Set_Num_Threads");
                if (dlerror()) throw std::runtime_error("Error: cannot load symbol 'MKL_Set_Num_Threads'");
                dlclose(handle);
            }
            fptr(nt);
        }
    private:
        int nt;
    };

    class mkl_init {
    private:
        mkl_init() : default_nt(0) {
            if (ambient::isset("AMBIENT_MKL_PARALLEL")) {
                if (ambient::isset("MKL_NUM_THREADS"))
                    default_nt = ambient::getint("MKL_NUM_THREADS");
                else
                    default_nt = 1;
                if (ambient::isset("AMBIENT_VERBOSE"))
                    std::cout << "ambient: selective mkl threading (" << default_nt << ")\n\n";
                mkl_parallel(1);
            }
        }
    public:
        template<class T>
        struct weak_instance {
            static mkl_init w;
        };
        int default_nt;
    };

    template<class T>
    mkl_init mkl_init::weak_instance<T>::w;

    inline int get_default_nt() {
        return mkl_init::weak_instance<void>::w.default_nt;
    }
}

#endif
