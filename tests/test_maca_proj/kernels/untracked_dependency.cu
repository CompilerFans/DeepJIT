// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
// `third_party/` is not a configured include prefix, so the parser ignores it;
// only the `-I` flag (and the compiler-option hash) distinguishes checkouts.
#include <third_party/dependency_value.cuh>

extern "C" __global__ void untracked_dependency_kernel(int* output, const int input) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = input + kThirdPartyDependencyValue;
}
