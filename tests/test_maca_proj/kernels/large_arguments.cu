// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
// Large parameter block: 1 KiB by value, eighteen 128-byte opaque arguments,
// an indirect payload, and a null pointer. Keep the fold inline (private limit).
struct OpaqueArgument {
    unsigned char bytes[128];
};

struct LargeArgumentPayload {
    unsigned long long values[128];
};

struct IndirectArgumentPayload {
    int values[8];
};

#define DJ_FOLD_OPAQUE(checksum, argument)                                    \
    for (int fold_index = 0; fold_index < 128; ++fold_index)                  \
        checksum = (checksum ^ (argument).bytes[fold_index]) * 1099511628211ull;

extern "C" __global__ void large_arguments_kernel(
    unsigned long long* output,
    const LargeArgumentPayload payload,
    OpaqueArgument opaque_0, OpaqueArgument opaque_1, OpaqueArgument opaque_2, OpaqueArgument opaque_3,
    OpaqueArgument opaque_4, OpaqueArgument opaque_5, OpaqueArgument opaque_6, OpaqueArgument opaque_7,
    OpaqueArgument opaque_8, OpaqueArgument opaque_9, OpaqueArgument opaque_10, OpaqueArgument opaque_11,
    OpaqueArgument opaque_12, OpaqueArgument opaque_13, OpaqueArgument opaque_14, OpaqueArgument opaque_15,
    OpaqueArgument opaque_16, OpaqueArgument opaque_17,
    const IndirectArgumentPayload indirect_payload,
    const void* optional_pointer) {
    if (blockIdx.x != 0 or threadIdx.x != 0)
        return;

    unsigned long long checksum = 1469598103934665603ull;
    for (int index = 0; index < 128; ++index)
        checksum = (checksum ^ payload.values[index]) * 1099511628211ull;
    DJ_FOLD_OPAQUE(checksum, opaque_0)
    DJ_FOLD_OPAQUE(checksum, opaque_1)
    DJ_FOLD_OPAQUE(checksum, opaque_2)
    DJ_FOLD_OPAQUE(checksum, opaque_3)
    DJ_FOLD_OPAQUE(checksum, opaque_4)
    DJ_FOLD_OPAQUE(checksum, opaque_5)
    DJ_FOLD_OPAQUE(checksum, opaque_6)
    DJ_FOLD_OPAQUE(checksum, opaque_7)
    DJ_FOLD_OPAQUE(checksum, opaque_8)
    DJ_FOLD_OPAQUE(checksum, opaque_9)
    DJ_FOLD_OPAQUE(checksum, opaque_10)
    DJ_FOLD_OPAQUE(checksum, opaque_11)
    DJ_FOLD_OPAQUE(checksum, opaque_12)
    DJ_FOLD_OPAQUE(checksum, opaque_13)
    DJ_FOLD_OPAQUE(checksum, opaque_14)
    DJ_FOLD_OPAQUE(checksum, opaque_15)
    DJ_FOLD_OPAQUE(checksum, opaque_16)
    DJ_FOLD_OPAQUE(checksum, opaque_17)
    for (int index = 0; index < 8; ++index)
        checksum = (checksum ^ static_cast<unsigned int>(indirect_payload.values[index])) * 1099511628211ull;
    output[0] = optional_pointer == nullptr ? checksum : 0;
}
