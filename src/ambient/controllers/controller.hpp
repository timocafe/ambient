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

#include "utils/rss.hpp"
#define STACK_RESERVE 65536

namespace ambient { namespace controllers {

    inline controller::~controller(){ 
        if(!chains->empty()) printf("Ambient:: exiting with operations still in queue!\n");
        this->clear();
    }

    inline controller::controller() : chains(&stack_m), mirror(&stack_s), clock(1) {}

    inline void controller::reserve(){
        this->stack_m.reserve(STACK_RESERVE);
        this->stack_s.reserve(STACK_RESERVE);
        this->garbage.reserve(STACK_RESERVE);
    }

    inline void controller::flush(){
        AMBIENT_SMP_ENABLE
        while(!chains->empty()){
            for(auto task : *chains){
                if(task->ready()){
                    AMBIENT_THREAD task->invoke();
                    for(auto d : task->deps) d->ready();
                    mirror->insert(mirror->end(), task->deps.begin(), task->deps.end());
                }else mirror->push_back(task);
            }
            chains->clear();
            std::swap(chains,mirror);
        }
        AMBIENT_SMP_DISABLE
        clock++;
        fence();
    }

    inline bool controller::empty(){
        return this->chains->empty();
    }

    inline void controller::clear(){
        this->garbage.clear();
        this->memory.reset();
    }

    inline bool controller::queue(functor* f){
        this->chains->push_back(f);
        return true;
    }

    inline bool controller::update(revision& r){
        if(r.assist.first != clock){
            r.assist.first = clock;
            return true;
        }
        return false;
    }

    inline void controller::sync(revision* r){
        if(is_serial()) return;
        if(model::common(r)) return;
        if(model::local(r)) set<revision>::spawn(*r);
        else get<revision>::spawn(*r);
    }

    inline void controller::lsync(revision* r){
        if(model::common(r)) return;
        if(!model::local(r)) get<revision>::spawn(*r);
    }

    inline void controller::rsync(revision* r){
        if(model::common(r)) return;
        if(model::local(r)) set<revision>::spawn(*r);
        else get<revision>::spawn(*r); // assist
    }

    inline void controller::lsync(transformable* v){
        if(is_serial()) return;
        set<transformable>::spawn(*v);
    }

    inline void controller::rsync(transformable* v){
        get<transformable>::spawn(*v);
    }

    template<typename T> void controller::collect(T* o){
        this->garbage.push_back(o);
    }

    inline void controller::squeeze(revision* r) const {
        if(r->valid() && !r->referenced() && r->locked_once()){
            if(r->spec.region == region_t::standard){
                ambient::memory::free(r->data, r->spec);
                r->spec.region = region_t::delegated;
            }
        }
    }

    inline void controller::touch(const history* o){
        model::touch(o);
    }

    inline void controller::use_revision(history* o){
        model::use_revision(o);
    }

    template<locality L, typename G>
    void controller::add_revision(history* o, G g){
        model::add_revision<L>(o, g);
    }

    inline int controller::get_tag_ub() const {
        return channel.tag_ub;
    }

    inline rank_t controller::get_rank() const {
        return channel.rank;
    }

    inline rank_t controller::get_shared_rank() const {
        return get_num_procs();
    }

    inline bool controller::is_serial() const {
        return (get_num_procs() == 1);
    }
        
    inline bool controller::verbose() const {
        return (get_rank() == 0);
    }

    inline void controller::fence() const {
        channel.barrier();
    }

    inline void controller::check_mem() const {
        double peak_size = (double)getPeakRSS();
        double avail_size = (double)getRSSLimit();
        if(peak_size / avail_size < 0.9) return;
        double current_size = (double)getCurrentRSS();
        printf("R%d: current: %.2f%%; peak: %.2f%%\n", get_rank(), (current_size/avail_size)*100, (peak_size/avail_size)*100);
    }

    inline int controller::get_num_procs() const {
        return channel.dim();
    }

    inline typename controller::channel_type & controller::get_channel(){
        return channel;
    }

} }

#undef STACK_RESERVE
