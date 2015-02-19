/*
 * Copyright Institute for Theoretical Physics, ETH Zurich 2014.
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

#ifndef AMBIENT_UTILS_TIMINGS
#define AMBIENT_UTILS_TIMINGS
#include "ambient/ambient.hpp"

namespace ambient {

    void sync();

    #if defined(__APPLE__) && defined(AMBIENT_OMP)
    struct time {
        static double start(){ return omp_get_wtime(); }
        static double stop() { return omp_get_wtime(); }
    };
    #elif defined(__APPLE__)
    struct time {
        static double start(){ return 0.; }
        static double stop() { return 0.; }
    };
    #else
    #define BILLION 0x3B9ACA00
    struct time {
        time(): thread_(pthread_self()){}
        double parse(timespec& t){
            return t.tv_sec+(((double)t.tv_nsec / (double)BILLION));
        }
        double start(){
            pthread_getcpuclockid(this->thread_,&this->cid_);
            struct timespec ts;
            clock_gettime(this->cid_, &ts);
            return parse(ts);
        }
        double stop(){
            struct timespec ts;
            clock_gettime(this->cid_, &ts);
            return parse(ts);
        }
        pthread_t thread_; 
        clockid_t cid_;
    };
    #endif

    class async_timer : public time {
    public:
        async_timer(std::string name): val(0.0), name(name), count(0){}
       ~async_timer(){ report(); }

        double get_time() const {
            return val;
        }
        void report(){
            std::cout << "R" << ambient::rank() << ": " << name << " " << val << ", count : " << count << "\n";
        }
        void reset(){
            this->val = 0;
        }
        void begin(){
            this->t0 = this->start();
        }    
        void end(){
            this->val += this->stop() - this->t0;
            count++;
        }
    private:
        double val, t0;
        unsigned long long count;
        std::string name;
    };

    class timer : public async_timer {
    public:
        timer(std::string name) : async_timer(name){}
        void begin(){
            ambient::sync();
            async_timer::begin();
        }    
        void end(){
            ambient::sync();
            async_timer::end();
        }
    };
}

#endif
