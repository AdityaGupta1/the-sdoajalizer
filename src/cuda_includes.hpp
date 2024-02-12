#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call)							                    \
  {								                                     	\
     cudaError_t err = call;                                            \
     if (err != cudaSuccess) {                                          \
       fprintf(stderr, "CUDA call (%s) failed with error: '%s (%s)' (%s:%u)\n", #call, cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__);\
       exit(EXIT_FAILURE);                                              \
     }                                                                  \
  }
