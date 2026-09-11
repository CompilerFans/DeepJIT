// A kernel with no body and no arguments: the case that uses it measures what
// a launch costs on the CPU, so any work the kernel did would be part of the
// measurement.  The CUDA harness carries the same fixture.
extern "C" __global__ void launch_overhead_kernel() {}
