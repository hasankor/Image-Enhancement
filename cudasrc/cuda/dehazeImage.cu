#include "dehazeImage.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "Utils.h"

__global__ void findMinChannel(const unsigned char* image, unsigned char* minChannel, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    unsigned char minVal = 255;

    for (int c = 0; c < channels; c++) {
        int idx = (y * width + x) * channels + c;
        if (image[idx] < minVal) {
            minVal = image[idx];
        }
    }

    minChannel[y * width + x] = minVal;
}

__global__ void boxFilter(const unsigned char* image, float* outputImage, int width, int height, int channels, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = (y * width + x) * channels;
    float sum = 0.0f;
    int count = 0;

    for (int kx = -kernelSize/2; kx <= kernelSize/2; kx++) {
        for (int ky = -kernelSize/2; ky <= kernelSize/2; ky++) {
            int px = x + kx;
            int py = y + ky;

            if (px < 0 || py < 0 || px >= width || py >= height) {
                continue;
            }

            int pidx = (py * width + px) * channels;
            for (int c = 0; c < channels; c++) {
                sum += image[pidx + c];
            }
            count++;
        }
    }

    outputImage[idx] = sum / count;
    outputImage[idx+1] = sum / count;
    outputImage[idx+2] = sum / count;
}

__global__ void computeTransmission(const unsigned char* minChannel, float* transmission, int width, int height, int kernelSize, float omega) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    float minVal = static_cast<float>(minChannel[idx]) / 255.0f;

    transmission[idx] = 1.0f - omega * minVal;
}

__global__ void dehazeImage(const unsigned char* image, const float* transmission, unsigned char* outputImage, int width, int height, int channels, float tMin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = (y * width + x) * channels;
    float t = fmaxf(transmission[y * width + x], tMin);
    float factor = 1.0f / t;

    for (int c = 0; c < channels; c++) {
        outputImage[idx+c] = fminf(factor * image[idx+c], 255.0f);
    }
}


void cudaDehazeImage(const cv::Mat &input, cv::Mat &output){

    // Create blank output image & calculate size of input and output
    output = cv::Mat(input.size(), input.type());


    unsigned char* imageData = input.data;
    int width = input.cols, height = input.rows, channels = input.channels();

    // Allocate memory for intermediate and output images
    unsigned char* darkChannel = (unsigned char*) malloc(width * height * sizeof(unsigned char));
    float* transmission = (float*) malloc(width * height * sizeof(float));
    unsigned char* outputData = (unsigned char*) malloc(width * height * channels * sizeof(unsigned char));

    // Allocate memory on the GPU
    unsigned char* d_imageData;
    GPU_HANDLE_ERROR(cudaMalloc((void**) &d_imageData, width * height * channels * sizeof(unsigned char)));

    unsigned char* d_darkChannel;
    GPU_HANDLE_ERROR(cudaMalloc((void**) &d_darkChannel, width * height * sizeof(unsigned char)));

    float* d_transmission;
    GPU_HANDLE_ERROR(cudaMalloc((void**) &d_transmission, width * height * sizeof(float)));

    unsigned char* d_outputData;
    GPU_HANDLE_ERROR(cudaMalloc((void**) &d_outputData, width * height * channels * sizeof(unsigned char)));

    // Copy input image to GPU
    GPU_HANDLE_ERROR(cudaMemcpy(d_imageData, imageData, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Compute dark channel
    dim3 blockSize(32, 32);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    findMinChannel<<<numBlocks, blockSize>>>(d_imageData, d_darkChannel, width, height, channels);

    // Compute transmission
    int kernelSize = 5;
    float omega = 0.95f;
    computeTransmission<<<numBlocks, blockSize>>>(d_darkChannel, d_transmission, width, height, kernelSize, omega);

    // Dehaze image
    float tMin = 0.1f;
    dehazeImage<<<numBlocks, blockSize>>>(d_imageData, d_transmission, d_outputData, width, height, channels, tMin);

    // Copy output image back to host
    GPU_HANDLE_ERROR(cudaMemcpy(outputData, d_outputData, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));


    //fill output image
    memcpy(output.data, outputData, width * height * channels * sizeof(unsigned char));

    // Free memory
    free(darkChannel);
    free(transmission);
    free(outputData);
    GPU_HANDLE_ERROR(cudaFree(d_imageData));
    GPU_HANDLE_ERROR(cudaFree(d_darkChannel));
    GPU_HANDLE_ERROR(cudaFree(d_transmission));
    GPU_HANDLE_ERROR(cudaFree(d_outputData));

}
