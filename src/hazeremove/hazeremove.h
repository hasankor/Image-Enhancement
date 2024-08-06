#ifndef HAZEREMOVE_H
#define HAZEREMOVE_H

#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

class HazeRemove
{
public:
    HazeRemove();

    void process(cv::Mat &pImage, cv::Mat &pOutput);

private:
    //median filtered dark channel
    void extractMedianDarkChannel(cv::Mat &pSource, int patch, cv::Mat &pOutputDarkChannel);

    //estimate airlight by the brightest pixel in dark channel (proposed by He et al.)
    void estimateA(cv::Mat &pDC, int &pOutputEstimate);

    //estimate transmission map
    void estimateTransmission(cv::Mat &pDCP, int ac, cv::Mat &pOutputTransmission);

    //dehazing foggy image
    void getDehazed(cv::Mat &pSource, cv::Mat &pTransmission, int al, cv::Mat &pOutputDehazed);

private:
    int mFrameCount;
    /*!
     * \brief mTempAirlight
     * temp airlight value
     */
    int mTempAirlight;
    /*!
     * \brief mCurrentAirlight
     * airlight value of current frame
     */
    int mCurrentAirlight;
    /*!
     * \brief mPreviousAirlight
     * airlight value of previous frame
     */
    int mPreviousAirlight;
    /*!
     * \brief mAlphaSmoothing
     * alpha smoothing
     */
    double mAlphaSmoothing;


};

#endif // HAZEREMOVE_H
