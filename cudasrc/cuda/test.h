#ifndef CUDATEST_H
#define CUDATEST_H


//#include "imageprocessor.h"
//#include <iostream>
//#include "radix_sort/sort.h"

////contrast streching
////        uint16_t tDiff = tMaxPixel - tMinPixel;

////        for (ushort y = 0; y < tFrame.rows; ++y) {
////            for (ushort x = 0; x < tFrame.cols; ++x) {
////                tOutputData[y * tFrame.cols + x] = std::round((tFrameData[y * tFrame.cols + x] - tMinPixel) * (65535 / tDiff));
////            }
////        }


///*!
// *working only grayscale image
// */
//template<typename T>
//__global__ void cudaStretching(T *input, T *output, int binDim, unsigned short cols, unsigned short minPixelval, unsigned short diffPixelVal)
//{
//    int outXind = blockIdx.x * blockDim.x + threadIdx.x;
//    int outYind = blockIdx.y * blockDim.y + threadIdx.y;

//    // Calculate x & y index of input binned pixels corresponding to current output pixel
//    int inXstart = outXind * binDim;
//    int inYstart = outYind * binDim;
//    for (int binY = inYstart; binY < (inYstart + binDim); ++binY) {
//        for (int binX = inXstart; binX < (inXstart + binDim); ++binX) {
//            int pixel_tid = binY * cols + binX;
//            output[pixel_tid] = (input[pixel_tid] - minPixelval) * (65535 / diffPixelVal);
//        }
//    }
//    __syncthreads();
//}
///*!
// * \brief binFilter
// * \param input
// * \param output
// * \info image deep must be 16u
// */
//void contrastStretch(const cv::Mat &input, cv::Mat &output)
//{
//    // 2X2 binning
//    int binDim = 1;

//    // Create blank output image & calculate size of input and output
//    output = cv::Mat(input.size(), input.type());
//    const int tInputSize = (input.rows * input.cols);

//    // Allocate memory in device
//    unsigned short *d_input, *d_output, *d_sortedOutput;
//    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_input, tInputSize * sizeof(unsigned short)));
//    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_output, tInputSize * sizeof(unsigned short)));
//    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_sortedOutput, tInputSize * sizeof(unsigned short)));

//    // Copy input image to device
//    GPU_HANDLE_ERROR(cudaMemcpy(d_input, input.ptr(), tInputSize * sizeof(unsigned short), cudaMemcpyHostToDevice));


//    unsigned short min = 0, max =0;

//    radix_sort(d_input, d_sortedOutput, tInputSize);

//    GPU_HANDLE_ERROR(cudaMemcpy(&min, &d_sortedOutput[10], sizeof(unsigned short), cudaMemcpyDeviceToHost));
//    GPU_HANDLE_ERROR(cudaMemcpy(&max, &d_sortedOutput[tInputSize - 10], sizeof(unsigned short), cudaMemcpyDeviceToHost));

//    std::cout << "Min value for image " << min << std::endl;
//    std::cout << "max value for image " << max << std::endl;

//    // Configure size of block and grid
//    const dim3 block(16, 16);
//    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y); // Additional block for rounding up

//    cudaStretching <unsigned short> <<<grid, block>>> (d_input, d_output, binDim, output.cols, min, max-min);

//    //    GPU_HANDLE_ERROR(cudaPeekAtLastError());

//    // Wait for all threads to finish
//    GPU_HANDLE_ERROR(cudaDeviceSynchronize());

//    // Copy output image from device back to host (cudaMemcpy is a blocking instruction)
//    GPU_HANDLE_ERROR(cudaMemcpy(output.ptr(), d_output, tInputSize * sizeof(unsigned short), cudaMemcpyDeviceToHost));

//    // Free device memory
//    GPU_HANDLE_ERROR(cudaFree(d_input));
//    GPU_HANDLE_ERROR(cudaFree(d_output));

//    // Write image to specified output_file path
//    //    cv::imwrite(output_file, output);
//}





/////////////////////////////////////////////////////////





//#include "imageprocessor.h"
//#include <iostream>
//#include "radix_sort/sort.h"

//typedef unsigned short Npp16u;

////contrast streching
////        uint16_t tDiff = tMaxPixel - tMinPixel;

////        for (ushort y = 0; y < tFrame.rows; ++y) {
////            for (ushort x = 0; x < tFrame.cols; ++x) {
////                tOutputData[y * tFrame.cols + x] = std::round((tFrameData[y * tFrame.cols + x] - tMinPixel) * (65535 / tDiff));
////            }
////        }


//template<typename T>
//__global__ void binCUDAKernel(T *input, T *output, int binDim, int outputWidth, int outputHeight, int inputWstep, int outputWstep, int nChannels)
//{
//    int outXind = blockIdx.x * blockDim.x + threadIdx.x;
//    int outYind = blockIdx.y * blockDim.y + threadIdx.y;

//    if ((outXind < outputWidth) && (outXind > outputWidth/2) && (outYind < outputHeight)) // Only run threads in output image coordinate range
//    {
//        if (nChannels == 1) // Test only for greyscale images
//        {
//            // Calculate x & y index of input binned pixels corresponding to current output pixel
//            int inXstart = outXind * binDim;
//            int inYstart = outYind * binDim;

//            // Perform binning on identified input pixels
//            float sum = 0;
//            for (int binY = inYstart; binY < (inYstart + binDim); binY++) {
//                for (int binX = inXstart; binX < (inXstart + binDim); binX++) {
//                    int input_tid = binY * inputWstep + binX;
//                    sum += float(input[input_tid]);
//                }
//            }

//            // Establish output thread index in current output pixel index
//            int output_tid = outYind * outputWstep + outXind;

//            // Assign binned pixel value to output pixel
//            output[output_tid] = static_cast<T>(sum / (binDim*binDim));
//        }
//         __syncthreads();
//    }
//}
///*!
// * \brief binFilter
// * \param input
// * \param output
// * \info image deep must be 16u
// */
//void contrastStretch(const cv::Mat &input, cv::Mat &output)
//{
//    // 2X2 binning
//    int binDim = 1;

//    // Create blank output image & calculate size of input and output
//    cv::Size outsize(input.size().width / binDim, input.size().height / binDim);
//    output = cv::Mat(outsize, input.type());
//    const int inputBytes = input.step * input.rows;
//    const int outputBytes = output.step * output.rows;

//    // Allocate memory in device
//    Npp16u *d_input, *d_output;
//    unsigned short *d_sortableInput, *d_sortadOutput;
//    GPU_HANDLE_ERROR(cudaMalloc<Npp16u>(&d_input, inputBytes));
//    GPU_HANDLE_ERROR(cudaMalloc<Npp16u>(&d_output, outputBytes));
//    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_sortableInput, (input.rows * input.cols) * sizeof(unsigned short)));
//    GPU_HANDLE_ERROR(cudaMalloc<unsigned short>(&d_sortadOutput, (input.rows * input.cols) * sizeof(unsigned short)));

//    // Copy input image to device
//    GPU_HANDLE_ERROR(cudaMemcpy(d_input, input.ptr(), (input.rows * input.cols) * sizeof(Npp16u), cudaMemcpyHostToDevice));
//    GPU_HANDLE_ERROR(cudaMemcpy(d_sortableInput, input.ptr(), (input.rows * input.cols) * sizeof(unsigned short), cudaMemcpyHostToDevice));


//    Npp16u min = 0, max =0;
//    auto start = std::clock();

//    radix_sort(d_sortableInput, d_sortadOutput, (input.rows * input.cols));

//    double gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
//    std::cout << "GPU time: " << gpu_duration << " s" << std::endl;

//    GPU_HANDLE_ERROR(cudaMemcpy(&min, &d_sortadOutput[10], sizeof(Npp16u), cudaMemcpyDeviceToHost));
//    GPU_HANDLE_ERROR(cudaMemcpy(&max, &d_sortadOutput[(input.rows * input.cols)-10], sizeof(Npp16u), cudaMemcpyDeviceToHost));

//    std::cout << "Min value for image " << min << std::endl;
//    std::cout << "max value for image " << max << std::endl;

//    int inputStep16 = input.step / sizeof(Npp16u);
//    int outputStep16 = output.step / sizeof(Npp16u);

//    // Configure size of block and grid
//    const dim3 block(16, 16);
//    const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y); // Additional block for rounding up

//    binCUDAKernel <Npp16u> <<<grid, block>>> (d_input, d_output, binDim, output.cols, output.rows, inputStep16, outputStep16, input.channels());

//    //    GPU_HANDLE_ERROR(cudaPeekAtLastError());

//    // Wait for all threads to finish
//    GPU_HANDLE_ERROR(cudaDeviceSynchronize());

//    // Copy output image from device back to host (cudaMemcpy is a blocking instruction)
//    GPU_HANDLE_ERROR(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost));

//    // Free device memory
//    GPU_HANDLE_ERROR(cudaFree(d_input));
//    GPU_HANDLE_ERROR(cudaFree(d_output));

//    // Write image to specified output_file path
//    //    cv::imwrite(output_file, output);
//}


#endif // CUDATEST_H
