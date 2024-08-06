#ifndef BADPIXELREPLACEMENT_H
#define BADPIXELREPLACEMENT_H

#include <QVector>
#include <opencv4/opencv2/core.hpp>

class BadPixelReplacement
{
public:
    BadPixelReplacement(QString pBadPixelFile, int pKernelSize = 3);


    void process8Bit(cv::Mat &pSource, cv::Mat &pDestination);
    void process16Bit(cv::Mat &pSource, cv::Mat &pDestination);
    void processVec(cv::Mat &pSource, cv::Mat &pDestination);

    void applyBpr(ushort *pData, int pRows = 480, int pCols = 640);

private:
    void processMedianBlur(cv::Mat &pSource, cv::Mat &pDestination);

private:
    int mKernelSize;

    /*!
     * \brief mBadPixelContainer
     * @arg x, y coordinate
     */
    QVector<QPair<int, int>> mBadPixelContainer;
};

#endif // BADPIXELREPLACEMENT_H
