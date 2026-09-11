// A kernel whose parameter block is deliberately large: 1 KiB by value, then
// eighteen 128-byte opaque arguments by value, then an indirectly-passed
// payload and a null pointer.  The opaque arguments stand in for the CUDA
// harness's `CUtensorMap`s (MACA has no tensor map type) -- what is under test
// is the argument-passing ABI and the parameter block size, not what the bytes
// mean.  CUDA marks the by-value aggregates `__grid_constant__`; MACA has no
// such qualifier, which is itself part of what this case covers.
//
// NOTE: the arguments are folded straight out of the parameter block.  Copying
// them into a local array first puts ~2.3 KiB into per-thread *private* memory,
// and this MACA platform caps that at 4 KiB/thread, so the launch is refused
// with mcErrorMemoryValueTooLarge ("kernel request: 5 KB/thread") before the
// kernel ever runs.  Keep the fold inline.
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
