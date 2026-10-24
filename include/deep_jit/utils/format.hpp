#pragma once

// Use fmt instead of `<format>`: some host toolchains (notably the GCC 11
// libstdc++ behind MACA's mxcc) predate it.  Header-only; consumers supply the
// include path.
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include <fmt/format.h>
