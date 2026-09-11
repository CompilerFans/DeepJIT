#pragma once

#include <third_party/dependency_value.cuh>

extern "C" __global__ void tracked_to_untracked_dependency_kernel(int* output, const int input) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = input + kThirdPartyDependencyValue;
}
