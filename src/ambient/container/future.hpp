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

#ifndef AMBIENT_CONTAINER_FUTURE_HPP
#define AMBIENT_CONTAINER_FUTURE_HPP

namespace ambient {

    using namespace ambient::models::ssm;

    template <typename T>
    class future {
    private:
        template<typename S> 
        future& operator = (const S& v){ }
    public:
        typedef T value_type;

        void init(value_type v = T()){
            desc = new (ambient::memory::calloc<memory::cpu::fixed,sizeof_transformable()>()) transformable_value<T>(v);
            valid = true;
        }
        template<typename S>
        void reuse(future<S>& f){
            desc = (transformable*)f.desc; // unsafe - proper convertion should be done
            valid = f.valid;
            f.clear();
        }
       ~future(){ 
           if(desc) ambient::destroy(desc); 
       }
        explicit constexpr future(transformable* c): desc(c) { // kernel's inner usage (no desctruction)
        }
        template <typename FP, FP OP> 
        explicit future(transformable_expr<T,FP,OP>* c): desc(c), valid(false){
        }
        future(){ 
            init();  
        }
        future(double v){ 
            init(v);   
        }
        future(std::complex<double> v){ 
            init(v);
        }
        T get() const { // kernels only
            return desc->v;
        }
        void set(T desired){ // kernels only
            desc->v = desired;
        }
        T load() const {
            if(!valid){
                ambient::sync();
                valid = true;
            }
            return desc->eval();
        }
        operator T () const {
            return load();
        }
        future(const future& f){
            init(f.load()); /* important */
        }
        future(future&& f){
            reuse(f);
        }
        future& operator = (const future& f){
            desc->v = f.load();
            return *this;
        }
        future& operator = (future&& f){ 
            if(desc) ambient::destroy(desc);
            reuse(f);
            return *this;
        }
        template<typename S>
        future(const future<S>& f){
            init((T)f.load());
        }
        template<typename S> 
        future(future<S>&& f){
            reuse(f);
        }
        future& operator += (const future& r){
            valid &= r.valid;
            *desc += *r.desc;
            r.clear();
            return *this;
        }
        future& operator /= (const future& r){
            desc->v = load() / r.load();
            return *this;
        }
        const future<T>& unfold() const {
            assert(valid);
            return *this;
        }
        future<T>& unfold(){
            assert(valid); 
            valid = false;
            return *this;
        }
        void clear() const {
            desc = NULL; 
        }

        template <class Archive> void load(Archive & ar, const unsigned int version = 0){ }
        template <class Archive> void save(Archive & ar, const unsigned int version = 0) const { }
    private:
        template <typename F> friend class future;
        mutable bool valid;
    public:
        mutable transformable* desc;
    };

    template<typename T>
    future<T> operator + (const future<T>& l, const future<T>& r){
        transformable* a = l.desc; l.clear();
        transformable* b = r.desc; r.clear();
        return future<T>(new (ambient::memory::calloc<memory::cpu::fixed,sizeof_transformable()>()) 
                         transformable_expr<T, decltype(&op_plus<T>), op_plus>(a, b)
                        ); 
    }
    template<typename T>
    T operator / (const future<T>& l, const future<T>& r){
        return (l.load() / r.load());
    }
    template<typename T> 
    T operator += (T& a, const future<T>& r){
        return (a += r.load());
    }
    template<typename T>
    T operator / (double l, const future<T>& r){
        return (l / r.load());
    }
    template<typename T>
    T operator / (std::complex<double> l, const future<T>& r){
        return (l / r.load());
    }

    template<class T>
    std::ostream& operator << (std::ostream& os, const future<T>& obj){
        os << obj.load();
        return os;
    }
}

namespace ambient { namespace numeric {

    inline double sqrt(const future<double>& f){ 
        return std::sqrt(f.load());
    }
    template<typename T>
    const future<double>& real(const future<T>& f){
        return *(future<double>*)&f;
    }
    template<typename T> 
    std::vector<double> real(const std::vector<future<T> >& f){
        ambient::sync();
        int size = f.size();
        std::vector<double> res; res.reserve(size);
        for(size_t k = 0; k < size; ++k) res.push_back(std::real(f[k].load()));
        return res;
    }

} }

#endif
