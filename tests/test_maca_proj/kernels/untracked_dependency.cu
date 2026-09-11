// `third_party/` is not one of the configured include prefixes, so the parser
// ignores this dependency; only the `-I` flag that points at it (and therefore
// the compiler-option hash) distinguishes one checkout of it from another.
#include <third_party/dependency_value.cuh>

extern "C" __global__ void untracked_dependency_kernel(int* output, const int input) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = input + kThirdPartyDependencyValue;
}
