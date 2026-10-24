// A valid device binary with no kernel: the artifact is produced, and only
// the load is refused because there is no entry point to name.
extern "C" __device__ int no_kernel_device_symbol = 7;
