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

#ifndef AMBIENT_MEMORY_POOL
#define AMBIENT_MEMORY_POOL

#include <sys/mman.h>
#include "ambient/utils/mutex.hpp"
#include "ambient/memory/factory.hpp"
#include "ambient/memory/region.hpp"
#include "ambient/memory/data_bulk.h"
#include "ambient/memory/instr_bulk.h"

namespace ambient { namespace memory {

    struct standard {
        static void* malloc(size_t sz){ return std::malloc(sz); }
        static void free(void* ptr){ std::free(ptr);  }
        static region_t signature(){
            return region_t::standard;
        }
    };

    struct fixed {
        // note: singleton_pool contains implicit mutex (critical for gcc)
        template<size_t S> static void* malloc(){ return std::malloc(S);   } // boost::singleton_pool<fixed,S>::malloc(); 
        template<size_t S> static void* calloc(){ return std::calloc(1,S); } // boost::singleton_pool<fixed,S>::malloc(); 
        template<size_t S> static void free(void* ptr){ std::free(ptr);    } // boost::singleton_pool<fixed,S>::free(ptr);
    };

} }

namespace ambient { namespace pool {

    using ambient::memory::data_bulk;
    using ambient::memory::fixed;
    using ambient::memory::standard;

    struct descriptor {

        descriptor(size_t e, region_t r = region_t::standard) : extent(e), region(r), persistency(1), crefs(1) {}

        void protect(){
            if(!(persistency++)) region = region_t::standard;
        }
        void weaken(){
            if(!(--persistency)) region = region_t::bulk;
        }
        void reuse(descriptor& d){
            region   = d.region;
            d.region = region_t::delegated;
        }
        bool conserves(descriptor& p){
            assert(p.region != region_t::delegated && region != region_t::delegated);
            return (!p.bulked() || bulked());
        }
        bool bulked(){
            return (region == region_t::bulk);
        }
        size_t extent;
        region_t region;
        int persistency;
        int crefs;
    };

    template<class Memory>           static void* malloc(size_t sz){ return Memory::malloc(sz);            }
    template<class Memory, size_t S> static void* malloc()         { return Memory::template malloc<S>();  }
    template<class Memory, class  T> static void* malloc()         { return malloc<Memory, sizeof(T)>();   }
    template<class Memory>           static void* calloc(size_t sz){ return Memory::calloc(sz);            }
    template<class Memory, size_t S> static void* calloc()         { return Memory::template calloc<S>();  }
    template<class Memory, class  T> static void* calloc()         { return calloc<Memory, sizeof(T)>();   }
    template<class Memory>           static void free(void* ptr)   { return Memory::free(ptr);             }
    template<class Memory, size_t S> static void free(void* ptr)   { return Memory::template free<S>(ptr); }
    template<class Memory, class  T> static void free(void* ptr)   { return free<Memory, sizeof(T)>(ptr);  }

    template<class Memory>
    static void* malloc(descriptor& d){
        d.region = Memory::signature();
        return Memory::malloc(d.extent);
    }

    static void* malloc(descriptor& d){
        assert(d.region != region_t::delegated);
        if(d.region == region_t::bulk){
            #ifdef AMBIENT_USE_DATA_BULK
            if(!data_bulk::open() || d.extent > AMBIENT_IB*AMBIENT_IB*16){  // <-- handle it separately
                d.region = region_t::standard;
                return malloc<standard>(d.extent);
            }
            return malloc<data_bulk>(d.extent); 
            #else
            d.region = region_t::standard;
            return malloc<standard>(d.extent);
            #endif
        } else return malloc<standard>(d.extent);
    }
    static void free(void* ptr, descriptor& d){ 
        if(ptr == NULL || d.region == region_t::delegated) return;
        if(d.region == region_t::bulk) free<data_bulk>(ptr);
        else free<standard>(ptr);
    }

} }

#endif
