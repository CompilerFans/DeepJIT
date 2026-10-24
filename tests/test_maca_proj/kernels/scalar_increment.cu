// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
extern "C" __global__ void increment_kernel(int* output, const int input) {
    output[0] = input + 1;
}
