// The kernel itself lives in a *tracked* header, which in turn includes an
// untracked one: the parser hashes the tracked file but stops at the untracked
// include, so only the compiler-option hash can separate the two checkouts.
#include <test_maca/tracked_to_untracked_dependency.hpp>
