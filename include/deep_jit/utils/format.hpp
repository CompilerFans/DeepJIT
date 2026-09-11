#pragma once

// DeepJIT formats through fmt instead of `<format>`.
//
// DeepJIT is a header-only library that gets embedded into a host extension, so
// its standard-library requirements become the *consumer's* requirements.  C++20
// `<format>` is not universally available in this project's deployment
// toolchains: MACA's `mxcc` drives the system GCC 11, whose libstdc++ predates
// `<format>` entirely (`std::format` there is a hard "no member named 'format'
// in namespace 'std'", not a missing transitive include).  fmt implements the
// same formatting language -- the same `{}` mini-language, the same output -- on
// any C++20 compiler, so it removes that floor without changing behaviour.
//
// fmt is used header-only, matching DeepJIT's own header-only design: the
// consumer supplies an include path to fmt (see README, "Integration") and
// nothing extra has to be linked.  That does mean FMT_HEADER_ONLY leaks to the
// rest of the consumer's translation unit; that is harmless unless the consumer
// also links a separately compiled fmt, which would then produce duplicate
// symbols.
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include <fmt/format.h>
