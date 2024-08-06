#ifndef CUDADEHAZEIMAGE_H
#define CUDADEHAZEIMAGE_H

#include <opencv2/core.hpp>

void cudaDehazeImage(const cv::Mat &input, cv::Mat &output);

#endif // CUDADEHAZEIMAGE_H
