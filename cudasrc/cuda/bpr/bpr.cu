#include "bpr.h"
#include "../Utils.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

template<typename TypeSrc>
__device__ void sortArray(TypeSrc *pInput, const int pLenght){
    for(int i = 0; i < pLenght - 1; ++i){
        for(int j = i + 1; j < pLenght; ++j){
            if(pInput[i] > pInput[j]){
                pInput[i] = pInput[i] ^ pInput[j];
                pInput[j] = pInput[i] ^ pInput[j];
                pInput[i] = pInput[i] ^ pInput[j];
            }
        }
    }
}

__global__ void bad_pixel_correction(unsigned short *input, unsigned short *output, const int rows, const int cols, const int kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int fromX = ((x - kernel) > 0 )   ? (x - kernel) : x;
    int toX   = (rows > (x + kernel)) ? (x + kernel) : rows;

    int fromY = ((y - kernel) > 0 )   ? (y - kernel) : y;
    int toY   = (cols > (y + kernel)) ? (y + kernel) : cols;


    unsigned short tBadPixelData = input[ x * cols + y];

    unsigned short neighborhood[50] = {0};
    int count = 0;

    for( int nx = fromX; nx < toX; ++nx){
        for( int ny = fromY; ny < toY; ++ny){
            unsigned short tPixelValue = input[ nx * cols + ny];
            if((tPixelValue != tBadPixelData) && (tPixelValue != 0) && (tPixelValue != USHRT_MAX)){
                neighborhood[count] = tPixelValue;
                ++count;
            }
        }
    }
    if(count > 0){
        sortArray<unsigned short>(neighborhood, count);
        output[ x * cols + y] = neighborhood[count / 2];
        __syncthreads();
    }

}

void badPixelReplacement(const unsigned short *pInput, unsigned short *pOutput, const int rows, const int cols){

    const static int kernel = 3;

    unsigned short *d_input, *d_output;

    GPU_HANDLE_ERROR(cudaMalloc(&d_input, rows * cols * sizeof(unsigned short)));
    GPU_HANDLE_ERROR(cudaMalloc(&d_output, rows * cols * sizeof(unsigned short)));

    GPU_HANDLE_ERROR(cudaMemcpy(d_input, pInput, rows * cols * sizeof(unsigned short), cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y);

    bad_pixel_correction<<<gridDim, blockDim>>>(d_input, d_output, rows, cols, kernel);


    GPU_HANDLE_ERROR(cudaMemcpy(pOutput, d_output, rows * cols * sizeof(unsigned short), cudaMemcpyDeviceToHost));

    GPU_HANDLE_ERROR(cudaPeekAtLastError());

    // Wait for all threads to finish
    GPU_HANDLE_ERROR(cudaDeviceSynchronize());

    cudaFree(d_input);
    cudaFree(d_output);
}
