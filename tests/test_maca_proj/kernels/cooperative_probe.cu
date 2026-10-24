// Records the grid shape it ran with.  This covers the cooperative launch
// attribute itself; no `cooperative_groups.h` is available here.
extern "C" __global__ void cooperative_probe_kernel(int* output) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = static_cast<int>(gridDim.x);
}
