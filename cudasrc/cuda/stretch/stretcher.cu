#include "stretcher.h"
#include <iostream>
#include <chrono>


#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

//cuda
#include "../radix_sort/sort.h"
#include "../Utils.h"

template<typename TypeSrc, typename TypeDst>
__global__ void findAvgOfArray(const TypeSrc *pInput, TypeSrc *pOutput, const int pArraySize) {
    int idx = threadIdx.x;

    TypeDst sum = 0.0;
    for (int i = idx; i < pArraySize; i += BLOCK_DIM)
        sum += pInput[i];

    __shared__ TypeDst shArr[BLOCK_DIM];
    shArr[idx] = sum;
    __syncthreads();

    for (int size = BLOCK_DIM/2; size>0; size/=2) { //uniform
        if (idx<size)
            shArr[idx] += shArr[idx+size];
        __syncthreads();
    }
    if (idx == 0)
        *pOutput = shArr[0] / pArraySize;
}

template<typename Type>
__global__ void findMinMaxWithAvgAndRatio(const Type *pInput, const int pArraySize, const Type pAvg, const float pRatio, Type *pMinOutput, Type *pMaxOutput){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int shared_index = threadIdx.x;

    __shared__ Type data_shared_min[BLOCK_DIM];
    __shared__ Type data_shared_max[BLOCK_DIM];

    const unsigned short tMaxPixelValue = (pAvg * (1 + pRatio));
    const unsigned short tMinPixelValue = (pAvg * pRatio);

    // check index condition
    if(index < pArraySize){

        if((pInput[index] >= tMinPixelValue) && (pInput[index] <= tMaxPixelValue)){

            data_shared_min[shared_index] = pInput[index]; //pass values from global to shared memory
            __syncthreads();
            data_shared_max[shared_index] = pInput[index]; //pass values from global to shared memory

            for (unsigned int stride = BLOCK_DIM/2; stride > 0; stride >>= 1) {
                if(threadIdx.x <  stride){
                    if(data_shared_max[threadIdx.x] <  data_shared_max[threadIdx.x+stride]) data_shared_max[shared_index] = data_shared_max[shared_index+stride];
                    if(data_shared_min[threadIdx.x] >  data_shared_min[threadIdx.x+stride]) data_shared_min[shared_index] = data_shared_min[shared_index+stride];
                }
                __syncthreads();
            }
            if(threadIdx.x == 0  ){
                atomicMin((unsigned int *)(pMinOutput), (Type)data_shared_min[threadIdx.x ]);
                //min_device =10;
                __syncthreads();
                atomicMax((unsigned int *)(pMaxOutput), (Type)data_shared_max[threadIdx.x ]);
            }

        }
    }else{
        data_shared_min[shared_index] = 9999;
    }
}

template<typename TypeSrc, typename TypeDst>
__global__ void cudaMemCopy(TypeSrc* src, TypeDst* dest, int binDim, size_t cols) {
    int outXind = blockIdx.x * blockDim.x + threadIdx.x;
    int outYind = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate x & y index of input binned pixels corresponding to current output pixel
    int inXstart = outXind * binDim;
    int inYstart = outYind * binDim;
    for (int binY = inYstart; binY < (inYstart + binDim); ++binY) {
        for (int binX = inXstart; binX < (inXstart + binDim); ++binX) {
            int pixel_tid = binY * cols + binX;
            dest[pixel_tid] = src[pixel_tid];
        }
    }
    __syncthreads();
}

/*!
 *working only grayscale image
 */
template<typename T>
__global__ void cudaStretching(T *input, T *output, int binDim, unsigned short cols, unsigned short minPixelval, float ratioOfMaxWithDiff)
{
    int outXind = blockIdx.x * blockDim.x + threadIdx.x;
    int outYind = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate x & y index of input binned pixels corresponding to current output pixel
    int inXstart = outXind * binDim;
    int inYstart = outYind * binDim;

    for (int binY = inYstart; binY < (inYstart + binDim); ++binY) {
        for (int binX = inXstart; binX < (inXstart + binDim); ++binX) {
            int pixel_tid = binY * cols + binX;
            __syncthreads();
            output[pixel_tid] = (input[pixel_tid] - minPixelval) * ratioOfMaxWithDiff;
        }
    }
    __syncthreads();
}


void contrastStretch(const unsigned short * pInput, unsigned short *pOutput, const int pRows, const int pCols)
{
    //2X2 binning
    const static int binDim = 2;

    const int tInputSize = pRows * pCols;

    // Allocate memory in device
    unsigned short *d_input, *d_output, *d_avg_output, *d_min_output, *d_max_output;
//    unsigned int *d_inputInt, *d_sortedOutput;

//    auto start = std::chrono::high_resolution_clock::now();
    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_input, tInputSize * sizeof(unsigned short)));
    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_output, tInputSize * sizeof(unsigned short)));
//    GPU_HANDLE_ERROR(cudaMalloc<unsigned int>(&d_inputInt, tInputSize * sizeof(unsigned int)));
//    GPU_HANDLE_ERROR(cudaMalloc<unsigned int>(&d_sortedOutput, tInputSize * sizeof(unsigned int)));
    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_avg_output, sizeof(unsigned short)));
    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_min_output, sizeof(unsigned short)));
    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_max_output, sizeof(unsigned short)));

    GPU_HANDLE_ERROR(cudaMemset(d_min_output, std::numeric_limits<ushort>::max(), sizeof(unsigned short)));
    GPU_HANDLE_ERROR(cudaMemset(d_max_output, std::numeric_limits<ushort>::min(), sizeof(unsigned short)));

//    std::cout<< "Duration Of cudaMalloc MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;


    // Copy input image to device
//    start = std::chrono::high_resolution_clock::now();
    GPU_HANDLE_ERROR(cudaMemcpy(d_input, pInput, tInputSize * sizeof(unsigned short), cudaMemcpyHostToDevice));
//    std::cout<< "Duration Of cudaMemcpy MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;

    // Configure size of block and grid
    const static dim3 block(32, 32);
    const static dim3 grid((pCols + block.x - 1) / block.x, (pRows + block.y - 1) / block.y); // Additional block for rounding up


//    start = std::chrono::high_resolution_clock::now();
    findAvgOfArray<unsigned short, unsigned long long><<<1, BLOCK_DIM>>>(d_input, d_avg_output, tInputSize);
    unsigned short tAvg = 0.0;
    GPU_HANDLE_ERROR(cudaMemcpy(&tAvg, d_avg_output, sizeof(unsigned short), cudaMemcpyDeviceToHost));

    dim3 numThreads(BLOCK_DIM);
    dim3 numBlocks(tInputSize/BLOCK_DIM + (tInputSize % BLOCK_DIM == 0 ? 0 : 1));

    findMinMaxWithAvgAndRatio<unsigned short><<<numBlocks, numThreads>>>(d_input, tInputSize, tAvg, 0.5, d_min_output, d_max_output);

    ushort tMaxPixel2 = 0;
    ushort tMinPixel2 = 0;

    GPU_HANDLE_ERROR(cudaMemcpy(&tMinPixel2, d_min_output, sizeof(unsigned short), cudaMemcpyDeviceToHost));
    GPU_HANDLE_ERROR(cudaMemcpy(&tMaxPixel2, d_max_output, sizeof(unsigned short), cudaMemcpyDeviceToHost));

//    std::cout <<"AVG::" << tAvg << std::endl;
//    std::cout <<"tMinPixel2::" << tMinPixel2 << std::endl;
//    std::cout <<"tMaxPixel2::" << tMaxPixel2 << std::endl;

//    std::cout<< "Duration Of Find SUM, MIN, MAX MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;


//    start = std::chrono::high_resolution_clock::now();
//    cudaMemCopy<unsigned short, unsigned int> <<<grid, block>>> (d_input, d_inputInt, binDim, pCols);
//    std::cout<< "Duration Of Copy MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;

//    start = std::chrono::high_resolution_clock::now();
//    radix_sort(d_inputInt, d_sortedOutput, tInputSize);
//    std::cout<< "Duration Of Sort MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;

    unsigned short min = tMinPixel2, max =tMaxPixel2;
//    GPU_HANDLE_ERROR(cudaMemcpy(&min, &d_inputInt[10], sizeof(unsigned short), cudaMemcpyDeviceToHost));
//    GPU_HANDLE_ERROR(cudaMemcpy(&max, &d_inputInt[tInputSize - 10], sizeof(unsigned short), cudaMemcpyDeviceToHost));

//    start = std::chrono::high_resolution_clock::now();
    cudaStretching <unsigned short> <<<grid, block>>> (d_input, d_output, binDim, pCols, min, (UINT16_MAX / (max-min)));
//    std::cout<< "Duration Of Stretching MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;

    GPU_HANDLE_ERROR(cudaPeekAtLastError());

    // Wait for all threads to finish
    GPU_HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy output image from device back to host (cudaMemcpy is a blocking instruction)
    GPU_HANDLE_ERROR(cudaMemcpy(pOutput, d_output, tInputSize * sizeof(unsigned short), cudaMemcpyDeviceToHost));

    // Free device memory
    GPU_HANDLE_ERROR(cudaFree(d_input));
    GPU_HANDLE_ERROR(cudaFree(d_output));
//    GPU_HANDLE_ERROR(cudaFree(d_inputInt));
//    GPU_HANDLE_ERROR(cudaFree(d_sortedOutput));
    GPU_HANDLE_ERROR(cudaFree(d_avg_output));
    GPU_HANDLE_ERROR(cudaFree(d_min_output));
    GPU_HANDLE_ERROR(cudaFree(d_max_output));
}
