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

#ifndef AMBIENT_CONTROLLERS_SSM_SCOPE
#define AMBIENT_CONTROLLERS_SSM_SCOPE

namespace ambient {

    class scope {
    public:
        typedef std::vector<int> container;
        typedef container::const_iterator const_iterator;
        static const_iterator balance(int k, int max_k);
        static const_iterator permute(int k, const std::vector<int>& s);
        static bool nested();
        static bool local();
        static scope& top();
        static size_t size();
        static const_iterator begin();
        static const_iterator end();
       ~scope();
        scope(const_iterator first, const_iterator last);
        container provision;
    };

    class actor {
    protected:
        typedef models::ssm::model model_type;
        typedef controllers::ssm::controller controller_type;
        actor(){}
    public:
       ~actor();
        actor(scope::const_iterator it);
        actor(actor_t type);
        bool remote() const;
        bool local()  const;
        bool common() const;
        rank_t which()  const;
        actor_t type;
        bool dry;
        int factor;
        int round;
        rank_t rank;
        ambient::locality state;
        controller_type* controller;
    };

    class base_actor : public actor {
    public:
        typedef typename actor::model_type model_type;
        base_actor();
        void set(rank_t r);
        void set(scope::const_iterator it);
        void schedule();
        void intend_read(models::ssm::revision* o);
        void intend_write(models::ssm::revision* o);
        mutable std::vector<rank_t> stakeholders;
        mutable std::vector<int> scores;
    };

}

#endif