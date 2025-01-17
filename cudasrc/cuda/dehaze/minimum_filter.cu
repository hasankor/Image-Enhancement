#include "minimum_filter.h"
#include "../Utils.h"

__global__ void d_minfilter_x(float *src, float *dst, int width, int height, int r){
    int i, j;
    int mask, len, extra, num, head, rear;
    int bid, tid;
    bid = blockIdx.x;
    tid = threadIdx.x;
    //float *p, *q;
    __shared__ float g[MAX_SIZE];
    __shared__ float h[MAX_SIZE];

    mask = 2 * r + 1;
    len = width + 2 * r + mask - (width + 2 * r) % mask;
    extra = len - width - r;
    num = len / mask;


    if (bid<height){
        //p = src + bid * width;
        //q = dst + bid * width;

        for (i = tid; i<r; i += BLOCKSIZE){
            g[i] = MAX_VALUE;
            h[i] = MAX_VALUE;
        }
    __syncthreads();
        for (i = tid; i<width; i += BLOCKSIZE){
            g[r + i] = src[bid * width + i];
            h[r + i] = src[bid * width + i];
        }
    __syncthreads();
        for (i = tid; i<extra; i += BLOCKSIZE){
            g[r + width + i] = MAX_VALUE;
            h[r + width + i] = MAX_VALUE;
        }

        __syncthreads();
        for (i = tid; i<num; i += BLOCKSIZE){
            head = i*mask;
            rear = (i + 1)*mask - 1;
            for (j = head + 1; j<(head + mask); j++){
                g[j] = (g[j - 1] < g[j]) ? g[j - 1] : g[j];
                h[rear - j + head] = (h[rear - j + head + 1] < h[rear - j + head]) ? h[rear - j + head + 1] : h[rear - j + head];
            }
        }
        __syncthreads();
        for (i = tid; i<width; i += BLOCKSIZE)
            dst[bid * width + i] = (g[i + r + r] > h[i]) ? h[i] : g[i + r + r];
        //dst[bid * width + i] = g[i+r];
    }
}

__global__ void d_minfilter_y(float *src, float *dst, int width, int height, int r){
    int i, j;
    int mask, len, extra, num, head, rear;
    int bid, tid;
    bid = blockIdx.x;
    tid = threadIdx.x;
    //float *p, *q;

    extern __shared__ float g[ ];
    extern __shared__ float h[ ];

    mask = 2 * r + 1;
    len = height + 2 * r + mask - (height + 2 * r) % mask;
    extra = len - height - r;
    num = len / mask;


    if (bid<width){
        //p = src + bid;
        //q = dst + bid;
        for (i = tid; i<height; i += BLOCKSIZE){
            g[r + i] = src[i*width + bid];
            h[r + i] = src[i*width + bid];
        }

        for (i = tid; i<r; i += BLOCKSIZE){
            g[i] = MAX_VALUE;
            h[i] = MAX_VALUE;
        }


        for (i = tid; i<extra; i += BLOCKSIZE){
            g[r + height + i - 1] = MAX_VALUE;
            h[r + height + i - 1] = MAX_VALUE;
        }

        for (i = tid; i<num; i += BLOCKSIZE){
            head = i*mask;
            rear = (i + 1)*mask - 1;
            for (j = head + 1; j<head + mask; j++){
                g[j] = (g[j - 1] < g[j]) ? g[j - 1] : g[j];
                h[rear - j + head] = (h[rear - j + head + 1] < h[rear - j + head]) ? h[rear - j + head + 1] : h[rear - j + head];
            }
        }
        __syncthreads();
        for (i = tid; i<height; i += BLOCKSIZE)
            dst[i*width + bid] = (g[i + r + r] > h[i]) ? h[i] : g[i + r + r];

    }
}
