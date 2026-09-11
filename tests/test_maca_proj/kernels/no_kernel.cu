// Compiles to a perfectly valid device binary that contains no kernel: the
// other half of the kernel-count contract from `multiple_kernels.cu`.  The
// artifact is produced (so `kernel.cu` and `kernel.devbin` are both written),
// and only the *load* is refused, because there is no entry point to name.
extern "C" __device__ int no_kernel_device_symbol = 7;
