int main() {

    FILE* f = fopen("weights.bin", "rb");

    // example: read embeddings
    fread(embedding, sizeof(float), 16 * 8, f);

   // read positional encoding
    fread(positional, sizeof(float), 16 * 8, f);

    load_weights();

    Tensor input = create_input();

    Tensor output = transformer_forward(input);

    print(output);
}