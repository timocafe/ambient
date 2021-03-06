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

#ifdef  AMBIENT_TRACKING
#ifndef AMBIENT_UTILS_OVERSEER
#define AMBIENT_UTILS_OVERSEER

namespace ambient {
    
    using ambient::models::ssm::history;
    using ambient::controllers::ssm::functor;

    class overseer {
    public:
        struct log {
            static std::ofstream& stream(){
                static std::ofstream ofs(std::string("log.")+std::to_string(ambient::rank()), std::ofstream::binary);
                return ofs;
            }
            static void begin(functor* op){
                stream() << "<div class=\"kernel " << op->name() << "\">" << op->name() << "\n";
            }
            static void end(functor* op){
                stream() << "</div>\n";
            }
            static void modify(history* o, functor* op){
                std::string title = o->label.empty() ? std::to_string(o->id) : o->label;
                stream() << "<div class=\"output\">" << o->label << "[" << o->id << "]</div>\n";
            }
            static void get(history* o, functor* op){
                std::string title = o->label.empty() ? std::to_string(o->id) : o->label;
                stream() << "<div class=\"import\">" << o->label << "[" << o->id << "]</div>\n";
            }
            static void renaming(history* o, const std::string& label){
                if(label.empty()) return;
                if(!ambient::models::ssm::model::remote(o->back())) return;
                std::string title = o->label.empty() ? std::to_string(o->id) : o->label;
                stream() << "<div class=\"rename\" data-operand=\"" << o->id << "\" data-from=\"" << title << "\" data-to=\"" << label << "\"></div>\n";
            }
            static void stop(){
                stream() << "<div class=\"cut\"></div>\n";
            }
            static void region(std::string label){
                stream() << "<div class=\"region\" data-title=\"" << label << "\"></div>\n";
            }
        };
        template<typename V>
        static void track(V& o, const std::string& label){
            log::renaming(o.ambient_rc.desc, label);
            o->label = label;
        }
    };

}

#endif
#endif
