#include "main.hpp"

#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    std::cout << "welcome to sdoaj's america" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count)
    {
        std::cerr << "error: GPU device number greater than device count" << std::endl;
        return -1;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::cout << "device name:         " << deviceProp.name << std::endl;
    std::cout << "compute capability:  " << major << "." << minor << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    return 0;
}