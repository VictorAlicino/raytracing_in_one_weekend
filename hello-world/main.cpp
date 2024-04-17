#include <iostream>

int main() {

    // Image

    int image_width = 1280;
    int image_height = 720;

    // Render

    // PPM image format
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    /*
    P3 Format Specifier
    image-width  image-height
    maximum value for each color (255 means 8-bit color, I wonder if I try 1024 what would happen in the final image)
    */

    for (int j = 0; j < image_height; ++j) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.0;

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    std::clog << "\rDone.                 \n";
}