// Records the grid shape it ran with.  This MACA install ships no
// `cooperative_groups.h`, so there is no `grid.sync()` to exercise here: what
// the cooperative case covers is the launch attribute itself -- that a
// cooperative launch is accepted, runs, and leaves the plain path working.
extern "C" __global__ void cooperative_probe_kernel(int* output) {
    if (blockIdx.x == 0 and threadIdx.x == 0)
        output[0] = static_cast<int>(gridDim.x);
}
