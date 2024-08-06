#include "transpose.h"
#include "../Utils.h"

__global__ void transpose(float *idata, float *odata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    if (xIndex<width && yIndex<height){
        int index_in = xIndex + (yIndex)*width;
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    }
    __syncthreads();

    int xIndex_new = blockIdx.y * TILE_DIM + threadIdx.x;
    int yIndex_new = blockIdx.x * TILE_DIM + threadIdx.y;
    if (xIndex_new<height && yIndex_new<width){
        //for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
        {
            int index_out = xIndex_new + (yIndex_new)*height;
            //odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
            odata[index_out] = tile[threadIdx.x][threadIdx.y];
        }
    }
}
