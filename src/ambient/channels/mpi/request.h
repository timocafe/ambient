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

#ifndef AMBIENT_CHANNELS_MPI_REQUEST
#define AMBIENT_CHANNELS_MPI_REQUEST

namespace ambient { namespace channels { namespace mpi {

    class request_impl : public memory::cpu::use_bulk_new<request_impl> {
    public:
        request_impl(void(*impl)(request_impl*), typename channel::scalar_type& v, rank_t target, int tag = 0);
        request_impl(void(*impl)(request_impl*), typename channel::block_type& r, rank_t target, int tag = 0);
        inline bool operator()();
        void* data;
        int extent;
        int target; // MPI_INT
        MPI_Request mpi_request;
        void(*impl)(request_impl*);
        bool once;
        int tag;
    };

    class request {
        typedef ambient::bulk_allocator<request_impl*> allocator;
    public:
        bool operator()();
        void operator &= (request_impl* r);
        void operator += (request_impl* r);
    private:
        std::vector<request_impl*,allocator> primaries;
        std::vector<request_impl*,allocator> callbacks;
    };

} } }

#endif
