// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#pragma once

// fmt replaces `<format>` for host toolchains that predate it.  The library is
// vendored at third-party/fmt; consumers add third-party/fmt/include to the
// include path (DEEP_JIT_FMT_ROOT and the test harness default there).
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include <fmt/format.h>
