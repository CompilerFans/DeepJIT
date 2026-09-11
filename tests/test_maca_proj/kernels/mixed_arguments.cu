// Exercises the kernel-argument ABI: several scalar widths in one signature,
// plus a raw pointer that the caller passes through `NoRefPtr` (which hands
// the pointer itself to the driver instead of its address).
extern "C" __global__ void mixed_arguments_kernel(float* output,
                                                  const int a,
                                                  const float b,
                                                  const double c,
                                                  const long long d,
                                                  const short e,
                                                  const unsigned char f,
                                                  const int* g) {
    output[0] = static_cast<float>(a) + b + static_cast<float>(c) + static_cast<float>(d) +
                static_cast<float>(e) + static_cast<float>(f) + static_cast<float>(g[0]);
}
