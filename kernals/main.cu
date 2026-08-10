#include "kernels.cu"
#include "tensor.h"

void load_weights() {
    FILE* f = fopen("weights.bin", "rb");

    fread(embedding, sizeof(float), 16 * 8, f);
    fread(positional, sizeof(float), 16 * 8, f);

    fclose(f);

    // First the embeddings and positional encodings when creating the input tensor, then we pass it to the GPU
    //Tensor input = create_input();

    // In the GPU we perform the transformer attention and feedforward operations, and get the output tensor
    // Tensor output = transformer_forward(input);
}





int main() {

    // Allocate arrays and give the add function parameters, then call the add kernel to test our setup. This is not part of the transformer implementation, just a sanity check.

    output = add<<<1, 256>>>(d_a, d_b, d_out, n);

    printf("%f\n", output[0]);
    
}