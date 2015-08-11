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

#ifndef AMBIENT
#define AMBIENT

#ifndef AMBIENT_DEFAULT_IB
#define AMBIENT_DEFAULT_IB           2048
#endif
#ifndef AMBIENT_INSTR_BULK_CHUNK
#define AMBIENT_INSTR_BULK_CHUNK     16777216 // 16 MB
#endif
#ifndef AMBIENT_DATA_BULK_CHUNK
#define AMBIENT_DATA_BULK_CHUNK      67108864 // 64 MB
#endif
#ifndef AMBIENT_COMM_BULK_CHUNK
#define AMBIENT_COMM_BULK_CHUNK      67108864 // 64 MB
#endif
#ifndef AMBIENT_MPI
#define AMBIENT_MPI                  MPI_THREAD_FUNNELED
#endif
#define MPI_DISABLE -1

// {{{ system includes
#if AMBIENT_MPI != MPI_DISABLE
#include <mpi.h>
#endif
#include <complex>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <limits>
#include <vector>
#include <stack>
#include <set>
#include <map>
#include <list>
#include <memory.h>
#include <stdarg.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <algorithm>
#include <execinfo.h>
#include <stdexcept>
#include <type_traits>
#include <functional>
#include <utility>
#include <atomic>
// }}}

#include "ambient/utils/dim2.hpp"
#include "ambient/utils/guard_once.hpp"
#include "ambient/utils/threading.hpp"
#include "ambient/utils/math.hpp"
#include "ambient/utils/rank_t.hpp"
#include "ambient/utils/io.hpp"

#include "ambient/memory/pool.hpp"
#include "ambient/memory/cpu/new.h"

#include "ambient/model/locality.h"
#include "ambient/model/functor.h"
#include "ambient/model/revision.h"
#include "ambient/model/history.h"
#include "ambient/model/transformable.h"

#ifdef MPI_VERSION
#include "ambient/channels/mpi/group.h"
#include "ambient/channels/mpi/channel.h"
#include "ambient/channels/mpi/request.h"
#include "ambient/channels/mpi/collective.h"
#else
#include "ambient/channels/nop/channel.h"
#endif

#include "ambient/controllers/collector.h"
#include "ambient/controllers/controller.h"
#include "ambient/controllers/meta.h"
#include "ambient/controllers/get.h"
#include "ambient/controllers/set.h"
#include "ambient/controllers/scope.h"
#include "ambient/controllers/actor.h"
#include "ambient/controllers/context.h"
#include "ambient/controllers/backbone.h"

#include "ambient/utils/auxiliary.hpp"

#include "ambient/memory/cpu/new.hpp"
#include "ambient/memory/cpu/data_bulk.hpp"
#include "ambient/memory/cpu/comm_bulk.hpp"
#include "ambient/memory/cpu/instr_bulk.hpp"

#include "ambient/model/revision.hpp"
#include "ambient/model/history.hpp"
#include "ambient/model/transformable.hpp"
#include "ambient/model/model.hpp"

#ifdef MPI_VERSION
#include "ambient/channels/mpi/group.hpp"
#include "ambient/channels/mpi/channel.hpp"
#include "ambient/channels/mpi/request.hpp"
#include "ambient/channels/mpi/collective.hpp"
#endif

#include "ambient/controllers/meta.hpp"
#include "ambient/controllers/get.hpp"
#include "ambient/controllers/set.hpp"
#include "ambient/controllers/collector.hpp"
#include "ambient/controllers/controller.hpp"
#include "ambient/controllers/scope.hpp"
#include "ambient/controllers/actor.hpp"
#include "ambient/controllers/context.hpp"
#include "ambient/controllers/backbone.hpp"

#include "ambient/interface/typed.hpp"
#include "ambient/interface/allocator.hpp"
#include "ambient/interface/kernel.hpp"
#include "ambient/interface/access.hpp"
#include "ambient/interface/lambda.hpp"

#include "ambient/container/proxy.hpp"
#include "ambient/container/block.hpp"
#endif
