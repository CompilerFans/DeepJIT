// Uses more dynamic shared memory than the default per-block limit, so the
// launch only succeeds when the shared-memory ceiling attribute was applied.
extern "C" __global__ void large_smem_kernel(int* output, const int input) {
    extern __shared__ int scratch[];
    scratch[threadIdx.x] = input;
    __syncthreads();
    if (threadIdx.x == 0)
        output[0] = scratch[0] + 1;
}
