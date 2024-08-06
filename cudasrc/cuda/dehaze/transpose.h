#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <cuda_runtime_api.h>

__global__ void transpose(float *idata, float *odata,  int width, int height);

#endif
