# Writing one simple kernel to test our setup. This is not used in the transformer implementation, just a sanity check to make sure we can compile and run CUDA code.

__global__ void add(float* a, float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}