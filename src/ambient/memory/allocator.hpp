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

#ifndef AMBIENT_MEMORY_ALLOCATOR_HPP
#define AMBIENT_MEMORY_ALLOCATOR_HPP

namespace ambient {

    template <class T>
    void* default_allocator<T>::alloc(memory::descriptor& spec) { return ambient::memory::malloc(spec); }

    template <class T>
    void* default_allocator<T>::calloc(memory::descriptor& spec){ void* m = alloc(spec); memset(m, 0, spec.extent); return m; }

    template <class T>
    void default_allocator<T>::free(void* ptr, memory::descriptor& spec){ ambient::memory::free(ptr, spec); }

    template <class T>
    T* bulk_allocator<T>::allocate(std::size_t n){ return (T*)ambient::memory::malloc<memory::instr_bulk>(n*sizeof(T)); }

    template <class T>
    void bulk_allocator<T>::deallocate(T* p, std::size_t n){ }

}

#endif
