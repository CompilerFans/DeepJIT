#ifndef TEST_OPTION
#define TEST_OPTION 0
#endif

extern "C" __global__ void option_kernel(int* output) {
    output[0] = TEST_OPTION;
}
