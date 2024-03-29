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

#ifndef AMBIENT_CONTROLLERS_CONTEXT_HPP
#define AMBIENT_CONTROLLERS_CONTEXT_HPP
#ifndef AMBIENT_SERIAL_COLLECTION

namespace ambient {

    using model::functor;

    inline context_mt::context_mt()
        : thread_context_lane(ambient::num_threads()), threaded_region(NULL)
    {
    }

    inline void context_mt::init(actor* base_actor) {
        for (thread_context& k : thread_context_lane) k.actors.push(base_actor);
    }

    inline typename context_mt::thread_context& context_mt::get() {
        return thread_context_lane[AMBIENT_THREAD_ID];
    }

    inline const typename context_mt::thread_context& context_mt::get() const {
        return thread_context_lane[AMBIENT_THREAD_ID];
    }

    inline bool context_mt::threaded() const {
        return (threaded_region != NULL);
    }

    inline void context_mt::sync() {
        for (int k = 1; k < thread_context_lane.size(); k++) {
            for (auto i : *thread_context_lane[k].controller.chains) thread_context_lane[0].controller.queue(i);
            thread_context_lane[k].controller.chains->clear();
        }
        for (auto& k : thread_context_lane) {
            k.controller.flush();
            k.controller.clear();
        }
    }

    inline void context_mt::diverge(int o) {
        get().offset = o;
    }

    inline void context_mt::fork(divergence_guard* guard) {
        assert(threaded_region == NULL);
        threaded_region = guard;
        for (auto& k : thread_context_lane) {
            k.actors.push(thread_context_lane[0].actors.top());
            k.scopes.push(thread_context_lane[0].scopes.top());
        }
        if (ambient::select().has_nested_actor())
            for (auto& k : thread_context_lane) {
                k.actors.push(new actor(*thread_context_lane[0].actors.top()));
                k.actors.top()->controller = &k.controller;
            } // otherwise assuming loop private actors
    }

    inline void context_mt::join() {
        threaded_region = NULL;
        if (ambient::select().has_nested_actor())
            for (auto& k : thread_context_lane) {
                k.actors.top()->controller = NULL; // avoiding destructor
                delete k.actors.top(); k.actors.pop();
            }
        for (auto& k : thread_context_lane) {
            k.actors.pop();
            k.scopes.pop();
        }
    }

    inline void context_mt::delay_transfer(controllers::meta* m) {
        threaded_region->transfers[get().offset].push_back(m);
    }

    inline context_mt::divergence_guard::divergence_guard(size_t length) : transfers(length) {
        ambient::select().fork(this);
    }
    inline context_mt::divergence_guard::~divergence_guard() {
        ambient::select().join();
        for (auto& transfers_part : transfers) for (auto& transfer : transfers_part) {
            if (!ambient::select().has_nested_actor()) ambient::select().get_base_actor().set(transfer->which);
            if (transfer->t == controllers::meta::type::set)
                controllers::set<model::revision>::spawn(transfer->r);
            else
                controllers::get<model::revision>::spawn(transfer->r);
        }
        for (auto& transfers_part : transfers) for (auto& transfer : transfers_part)
            if (transfer->t == controllers::meta::type::get) {
                for (auto d : transfer->deps) (transfer->r.generator.load())->queue(d);
            }
    }
}

#else

namespace ambient {

    inline void context_serial::init(actor* base_actor) {
        actors.push(base_actor);
    }

    inline void context_serial::sync() {
        controller.flush();
        controller.clear();
    }

    inline context_serial& context_serial::get() {
        return *this;
    }

    inline bool context_serial::threaded() const {
        return false;
    }

    inline void context_serial::delay_transfer(controllers::meta* m) {
    }

    inline void context_serial::fork(void*) {
    }

    inline void context_serial::join() {
    }

    inline void context_serial::diverge(int) {
    }

}

#endif
#endif
