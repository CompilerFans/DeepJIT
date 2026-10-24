// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
// Tracked include: angle-bracket form with the configured `test_maca/` prefix,
// so its contents enter the cache digest (see `test_include_dirs`).
#include <test_maca/tracked_include_value.hpp>

extern "C" __global__ void tracked_include_kernel(int* output, const int input) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = input + kTrackedOffset;
}
