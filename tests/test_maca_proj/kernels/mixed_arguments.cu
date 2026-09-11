// Exercises the kernel-argument ABI: several scalar widths in one signature,
// including a 1-byte `bool` (the width most likely to be sized wrong) and an
// 8-byte unsigned, plus a pointer the caller also passes through `NoRefPtr`.
//
// Every term is exactly representable in binary32, so the sum has to match bit
// for bit rather than within a tolerance.
extern "C" __global__ void mixed_arguments_kernel(float* output,
                                                  const int a,
                                                  const float b,
                                                  const double c,
                                                  const long long d,
                                                  const short e,
                                                  const unsigned char f,
                                                  const bool h,
                                                  const unsigned long long i,
                                                  const int* g) {
    output[0] = static_cast<float>(a) + b + static_cast<float>(c) + static_cast<float>(d) +
                static_cast<float>(e) + static_cast<float>(f) + (h ? 1.0f : 0.0f) +
                static_cast<float>(i) + static_cast<float>(g[0]);
}
