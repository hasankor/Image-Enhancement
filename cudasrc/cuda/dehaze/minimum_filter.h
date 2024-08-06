#ifndef MINIMUM_FILTER_H
#define MINIMUM_FILTER_H

#include <cuda_runtime_api.h>

__global__ void d_minfilter_x(float *src, float *dst, int width, int height, int r);

__global__ void d_minfilter_y(float *src, float *dst, int width, int height, int r);

#endif
