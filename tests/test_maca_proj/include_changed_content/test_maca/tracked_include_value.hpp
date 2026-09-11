#pragma once

// Tracked, and deliberately two levels deep: the value lives one include
// further down, so the cache digest can only change when the parser descends
// recursively.  `test_include_dirs` puts the one-byte fixture difference in
// `detail/tracked_offset.hpp` for exactly that reason.
#include <test_maca/detail/tracked_offset.hpp>
