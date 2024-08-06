#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "../Utils.h"


// process row
__device__ void d_boxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++)
    {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++)
    {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }

    // main loop
    for (int x = (r + 1); x < w - r; x++)
    {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++)
    {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

// process column
__device__ void d_boxfilter_y(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++)
    {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++)
    {
        t += id[(h-1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

__global__ void d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r);
}

void boxfilter(float *d_src, float *d_dest, float *d_temp, int height, int width, int radius)
{
    d_boxfilter_y_global<<<(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0 >>>(d_temp, d_dest, width, height, radius);
}
