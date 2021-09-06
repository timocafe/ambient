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

#ifndef AMBIENT_CONTAINER_NUMERIC_BINDINGS_TYPES
#define AMBIENT_CONTAINER_NUMERIC_BINDINGS_TYPES

namespace ambient {
    inline namespace numeric {
        namespace bindings {

            // {{{ overloaded convertion functions
            template <typename T, typename D, int IB>
            void convert(ambient::tiles<ambient::diagonal_matrix<T>, IB>& pm, const ambient::tiles<ambient::diagonal_matrix<D>, IB>& m) {
                for (size_t k = 0; k < m.data.size(); ++k)
                    ambient::numeric::kernels::template cast_double_complex<T, D>(pm[k], m[k]);
            }
            // }}}

            template <typename O, typename I> struct adaptor {};

            template <typename O, typename I> O cast(I const& input) {
                return adaptor<O, I>::convert(input);
            }

            template <typename T, typename D, int IB>
            struct adaptor< ambient::tiles<ambient::diagonal_matrix<T>, IB>, ambient::tiles<ambient::diagonal_matrix<D>, IB> > {
                static ambient::tiles<ambient::diagonal_matrix<T>, IB> convert(const ambient::tiles<ambient::diagonal_matrix<D>, IB>& m) {
                    ambient::tiles<ambient::diagonal_matrix<T>, IB> pm(num_rows(m), num_cols(m));
                    ambient::numeric::bindings::template convert<T, D>(pm, m);
                    return pm;
                }
            };

        }
    }
}

#endif
