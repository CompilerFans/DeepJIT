// Keeps a per-thread array alive across two loops, so the array cannot be
// promoted to registers and mxcc has to place it in the stack frame.  The
// `check_no_spills` / `check_no_local_memory` options must reject it.
extern "C" __global__ void local_frame_kernel(float* output, const float* input, const int size) {
    float buffer[2048];
    for (int index = 0; index < 2048; ++index)
        buffer[index] = input[index % size] * static_cast<float>(index);
    for (int index = 0; index < 2048; ++index)
        buffer[index] += buffer[(index * 7) % 2048];
    float sum = 0.0f;
    for (int index = 0; index < 2048; ++index)
        sum += buffer[index];
    output[0] = sum;
}
