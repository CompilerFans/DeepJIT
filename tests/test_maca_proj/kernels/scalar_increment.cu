extern "C" __global__ void increment_kernel(int* output, const int input) {
    output[0] = input + 1;
}
