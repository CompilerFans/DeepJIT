// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
// Kernel-argument ABI: mixed scalar widths, including a 1-byte `bool`, plus a
// pointer also passed through `NoRefPtr`.  All terms are exact in binary32.
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
