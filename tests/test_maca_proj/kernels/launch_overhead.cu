// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
// An empty kernel with no arguments: the case measures CPU launch cost, so any
// kernel work would be part of the measurement.
extern "C" __global__ void launch_overhead_kernel() {}
