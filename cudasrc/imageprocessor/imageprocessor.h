#ifndef CUDAIMAGEPROCESSOR_H
#define CUDAIMAGEPROCESSOR_H

#include <opencv2/core.hpp>

class CudaImageProcessor
{
public:
    CudaImageProcessor();

    void process(const cv::Mat &input, cv::Mat &output);
};

#endif // CUDAIMAGEPROCESSOR_H
