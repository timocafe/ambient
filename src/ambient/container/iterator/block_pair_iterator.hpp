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

#ifndef AMBIENT_CONTAINER_ITERATOR_BLOCK_PAIR_ITERATOR
#define AMBIENT_CONTAINER_ITERATOR_BLOCK_PAIR_ITERATOR

namespace ambient {

    template<int IB>
    class block_pair_iterator {
    public:
        block_pair_iterator(size_t first, size_t second, size_t size) 
        : first(first), second(second), lim(first+size){
            measure_step();
        }
        void operator++ (){
            first  += step;
            second += step;
            measure_step();
        }
        bool end(){
            return (first >= lim);
        }
        void measure_step(){
            step = std::min(std::min((IB*__a_ceil((first+1)/IB) - first), 
                                     (IB*__a_ceil((second+1)/IB) - second)),
                                     (lim-first));
        }
        size_t first;
        size_t second;
        size_t step;
        size_t lim;
    };

}

#endif
