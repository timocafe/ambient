/*
 * Ambient, License - Version 1.0 - May 3rd, 2012
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

#define UNDEFINED_RANK MPI_UNDEFINED

namespace ambient { namespace channels { namespace mpi {

    inline rank_t multirank::operator()() const {
        return this->world->rank;
    }

    inline rank_t multirank::operator()(const group* grp) const { 
        return grp->rank;
    }

    inline rank_t multirank::translate(rank_t rank, const group* source) const {
        if(source->depth == 0) return rank;
        return cast_to_parent(rank, source, this->world);
    }

    inline rank_t multirank::translate(rank_t rank, const group* source, const group* target) const {
        if(source->depth == target->depth) return rank;
        else if(target->depth < source->depth) return cast_to_parent(rank, source, target);
        else return cast_to_child(rank, source, target);
    }

    // query rank "i" inside a group (not ranks[i])
    inline rank_t multirank::cast_to_parent(rank_t rank, const group* source, const group* target) const {
        for(const group* i = source; i != target; i = i->parent){
            assert(rank < i->size);
            rank = i->ranks[rank];
        }
        return rank;
    }
 
    inline rank_t multirank::cast_to_child(rank_t rank, const group* source, const group* target) const {
        if(target == source) return rank;
        rank = cast_to_child(rank, source, target->parent);
        for(int i = 0; i < target->size; ++i)
            if(target->ranks[i] == rank) return i;
        return UNDEFINED_RANK;
    }

    inline bool multirank::belongs(const group* target) const {
        return (target->rank != UNDEFINED_RANK);
    }

    inline rank_t multirank::left_neighbor() const {
        return ((*this)()-1 + this->world->size) % this->world->size;
    }

    inline rank_t multirank::right_neighbor() const {
        return ((*this)()+1) % this->world->size;
    }

} } }