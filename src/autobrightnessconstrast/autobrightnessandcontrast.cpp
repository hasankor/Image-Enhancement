#include "autobrightnessandcontrast.h"

AutoBrightnessAndContrast::AutoBrightnessAndContrast()
{

}

std::tuple<cv::Mat, float, float> AutoBrightnessAndContrast::automaticBrightnessAndContrast14bit(cv::Mat pGrayImage, float pClipHistPercent) {

    cv::Mat tHist;
    int tHistSize = 16383;
    float tRange[] = { 0, 16383 };
    const float* tHistRange = { tRange };
    cv::calcHist(&pGrayImage, 1, nullptr, cv::Mat(), tHist, 1, &tHistSize, &tHistRange);

    std::vector<float> tAccumulator(tHistSize);
    tAccumulator[0] = tHist.at<float>(0);
    for (int index = 1; index < tHistSize; ++index) {
        tAccumulator[index] = tAccumulator[index - 1] + tHist.at<float>(index);
    }

    float tMaximum = tAccumulator.back();
    pClipHistPercent *= tMaximum / 16383.0;
    pClipHistPercent /= 2.0;

    int tMinimumGray = 0;
    while (tAccumulator[tMinimumGray] < pClipHistPercent) {
        ++tMinimumGray;
    }

    int tMaximumGray = tHistSize - 1;
    while (tAccumulator[tMaximumGray] >= tMaximum - pClipHistPercent) {
        --tMaximumGray;
    }

    float tAlpha = 16383.0f / (tMaximumGray - tMinimumGray);
    float tBeta = -tMinimumGray * tAlpha;

    cv::Mat tAutoResult;
    cv::convertScaleAbs(pGrayImage, tAutoResult, tAlpha, tBeta);

    return std::make_tuple(tAutoResult, tAlpha, tBeta);
}
