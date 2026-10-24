// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
extern "C" __global__ void entry_first(int* output, const int input) {
    output[0] = input + 11;
}

extern "C" __global__ void entry_second(int* output, const int input) {
    output[0] = input + 22;
}
