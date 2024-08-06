#include "imageprocessor.h"
#include "../cuda/stretch/stretcher.h"
#include "../cuda/bpr/bpr.h"
#include "../cuda/dehaze/dehaze.h"
#include "../cuda/dehazeImage.h"
#include <iostream>
#include <chrono>

CudaImageProcessor::CudaImageProcessor() {}

void CudaImageProcessor::process(const cv::Mat &input, cv::Mat &output)
{
    // Create blank output image & calculate size of input and output
    output = cv::Mat(input.size(), input.type());

    unsigned short *tInput = reinterpret_cast<unsigned short*>(input.data);

    unsigned short *tBPROutput = new unsigned short[input.rows * input.cols];
    unsigned short *tStretchOutput = new unsigned short[input.rows * input.cols];

    unsigned char *tDehazeOutput = new unsigned char[input.rows * input.cols * input.channels()];


    //bpr on image
    auto start = std::chrono::high_resolution_clock::now();
    badPixelReplacement(tInput, tBPROutput, input.rows, input.cols);
    std::cout<< "Duration Of BPR MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;

    //    //stretching on image
    //    start = std::chrono::high_resolution_clock::now();
    contrastStretch(tBPROutput, tStretchOutput, input.rows, input.cols);
    //    std::cout<< "Duration Of Contrast Stretching MicroSec:" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;


    //    //fill output image
    //    memcpy(output.data, tDehazeOutput, input.rows * input.cols * input.channels());

    //dehazeGpu
    //    dehazeGPU(reinterpret_cast<unsigned char*>(tStretchOutput), tDehazeOutput, input.channels(), input.cols, input.rows);

    //    //fill output image
    memcpy(input.data, tStretchOutput, input.rows * input.cols * input.channels());
    cudaDehazeImage(input, output);

    //fill output image
    memcpy(output.data, tDehazeOutput, input.rows * input.cols * input.channels());
}
