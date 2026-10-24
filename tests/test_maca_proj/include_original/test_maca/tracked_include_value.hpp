#pragma once

// Deliberately two levels deep: the digest can only change if the parser
// descends recursively.  The fixture difference lives in the nested header.
#include <test_maca/detail/tracked_offset.hpp>
