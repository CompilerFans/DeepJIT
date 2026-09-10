extern "C" __global__ void vector_add_kernel(float* output, const float* left, const float* right, const int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = left[index] + right[index];
}
