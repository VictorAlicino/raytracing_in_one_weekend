#include <iostream>

__global__
void render(int image_width, int image_height, int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < image_width && j < image_height) {
        auto r = double(i) / (image_width - 1);
        auto g = double(j) / (image_height - 1);
        auto b = 0.0;

        int ir = int(255.999 * r);
        int ig = int(255.999 * g);
        int ib = int(255.999 * b);

        int index = j * image_width + i;
        output[index * 3] = ir;
        output[index * 3 + 1] = ig;
        output[index * 3 + 2] = ib;
    }
}

int main() {
    // Image
    int image_width = 3840;
    int image_height = 2160;
    int image_size = image_width * image_height * 3; // 3 channels (RGB)

    // Allocate memory on the host for output
    int* output_host = new int[image_size];

    // Allocate memory on the device for output
    int* output_device;
    cudaMalloc(&output_device, image_size * sizeof(int));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);
    render<<<gridSize, blockSize>>>(image_width, image_height, output_device);

    // Copy result back to host
    cudaMemcpy(output_host, output_device, image_size * sizeof(int), cudaMemcpyDeviceToHost);

    // PPM image format
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // Output pixel values
    for (int i = 0; i < image_size; ++i) {
        std::cout << output_host[i] << ' ';
        if ((i + 1) % (image_width * 3) == 0)
            std::cout << '\n';
    }

    // Free device memory
    cudaFree(output_device);

    // Free host memory
    delete[] output_host;

    std::clog << "\rDone.                 \n";
    return 0;
}