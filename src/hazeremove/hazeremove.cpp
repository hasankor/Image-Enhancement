#include "hazeremove.h"
#include <iostream>

HazeRemove::HazeRemove()
    : mFrameCount(0)
    , mTempAirlight(0)
    , mCurrentAirlight(0)
    , mPreviousAirlight(0)
    , mAlphaSmoothing(0.05)
{

}

void HazeRemove::process(cv::Mat &pImage, cv::Mat &pOutput)
{
    cv::Mat tDarkChannel;
    cv::Mat tTransmisson;

    extractMedianDarkChannel(pImage, 5, tDarkChannel);
    estimateA(tDarkChannel, mCurrentAirlight);
    estimateTransmission(tDarkChannel, mCurrentAirlight, tTransmisson);

    if (mFrameCount == 1) //first frame, without airlight smoothing
    {
        mTempAirlight = mCurrentAirlight;

    } else{ //other frames, with airlight smoothing

        mPreviousAirlight = mTempAirlight;
        mTempAirlight = int(mAlphaSmoothing * double(mCurrentAirlight) + (1 - mAlphaSmoothing) * double(mPreviousAirlight));//airlight smoothing
    }

    getDehazed(pImage, tTransmisson, mTempAirlight, pOutput);

}

void HazeRemove::extractMedianDarkChannel(cv::Mat &pSource, int patch, cv::Mat &pOutputDarkChannel)
{
    cv::Mat tRgbmin = cv::Mat::zeros(pSource.rows, pSource.cols, CV_8UC1);
    cv::Vec3b tIntensity;

    for(int row = 0; row < pSource.rows; ++row)
    {
        for(int col = 0; col < pSource.cols; ++col)
        {
            tIntensity = pSource.at<cv::Vec3b>(row, col);
            tRgbmin.at<uchar>(row, col) = cv::min(cv::min(tIntensity.val[0], tIntensity.val[1]), tIntensity.val[2]);
        }
    }
    cv::medianBlur(tRgbmin, pOutputDarkChannel, patch);
}

void HazeRemove::estimateA(cv::Mat &pDC, int &pOutputEstimate)
{
    double tMinDC, tMaxDC;
    cv::minMaxLoc(pDC, &tMinDC, &tMaxDC);
    pOutputEstimate = tMaxDC;
    std::cout<<"estimated airlight is:" << pOutputEstimate << std::endl;
}

void HazeRemove::estimateTransmission(cv::Mat &pDCP, int ac, cv::Mat &pOutputTransmission)
{
    double w = 0.75;
    pOutputTransmission = cv::Mat::zeros(pDCP.rows, pDCP.cols, CV_8UC1);

    for (int row = 0; row < pDCP.rows; ++row)
    {
        for (int col = 0; col < pDCP.cols; ++col)
        {
            pOutputTransmission.at<uchar>(row, col) = (1 - w * (static_cast<cv::Scalar>(pDCP.at<uchar>(row, col))).val[0] / ac) * 255;
        }
    }
}

void HazeRemove::getDehazed(cv::Mat &pSource, cv::Mat &pTransmission, int al, cv::Mat &pOutputDehazed)
{
    double tmin = 0.1;
    double tmax;

    cv::Scalar tInttran;
    cv::Vec3b tIntsrc;
    pOutputDehazed = cv::Mat::zeros(pSource.rows, pSource.cols, CV_8UC3);

    for(int row = 0; row < pSource.rows; ++row)
    {
        for(int col = 0; col < pSource.cols; ++col)
        {
            tInttran = pTransmission.at<uchar>(row, col);
            tIntsrc = pSource.at<cv::Vec3b>(row, col);
            tmax = (tInttran.val[0]/255) < tmin ? tmin : (tInttran.val[0]/255);
            for(int k=0; k<3; ++k)
            {
                pOutputDehazed.at<cv::Vec3b>(row, col)[k] = cv::abs((tIntsrc.val[k] - al) / tmax + al) > 255 ? 255 : cv::abs((tIntsrc.val[k] - al) / tmax + al);
            }
        }
    }
}
