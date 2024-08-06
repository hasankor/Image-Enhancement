#include<stdio.h>
#include "dehaze_kernel.h"
#include "minimum_filter.h"
#include "transpose.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "../Utils.h"

__global__ void d_min_img_kernel(float *src, float *dst, int iWidth, int iHeight, int iChannel){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float r, g, b;
	if (id<iHeight*iWidth){
		r = src[id*iChannel + 0];
		g = src[id*iChannel + 1];
		b = src[id*iChannel + 2];
        dst[id] = (r < g) ? r : g;
        dst[id] = (dst[id]< b) ? dst[id] : b;
	}
}

void minfilter(float *d_fog, float *d_min_img, float *d_win_dark, float *d_temp, int width, int height, int channel, int radius){

	dim3 grid1(width / TILE_DIM + 1, height / TILE_DIM + 1);
	dim3 grid2(height / TILE_DIM + 1, width / TILE_DIM + 1);
	dim3 block(TILE_DIM, TILE_DIM);


	if (channel == 3)
	{
		d_min_img_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_fog, d_min_img, width, height, channel);
	}
	
    d_minfilter_x << <height, BLOCKSIZE>> >(d_min_img, d_temp, width, height, radius);
    transpose << <grid1, block >> >(d_temp, d_min_img, width, height);

    d_minfilter_y << <width, BLOCKSIZE >> >(d_min_img, d_temp, height, width, radius);
    transpose << <grid2, block >> >(d_temp, d_win_dark, height, width);

}
