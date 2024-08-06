#include "badpixelreplacement.h"
#include <opencv2/photo.hpp>
#include <QDebug>
#include <QFile>

BadPixelReplacement::BadPixelReplacement(QString pBadPixelFile, int pKernelSize)
    : mKernelSize(pKernelSize)
{
    QFile tFile(pBadPixelFile);
    if (!tFile.open(QIODevice::ReadOnly)) {
        qDebug() << "BadPixelReplacement::BadPixelReplacement " <<  tFile.errorString();
        return;
    }

    while (!tFile.atEnd()) {
        QString line = tFile.readLine();
        line.remove("\r").remove("\n");
        QStringList tCoordinateList;
        if(line.contains(";")){
            tCoordinateList = line.split(";");
        }else if(line.contains(",")) {
            tCoordinateList = line.split(",");
        }
        if(tCoordinateList.size() == 2){
            mBadPixelContainer.append(qMakePair(tCoordinateList[1].toUInt(), tCoordinateList[0].toUInt()));
        }
    }
}

void BadPixelReplacement::processMedianBlur(cv::Mat &pSource, cv::Mat &pDestination)
{
    cv::medianBlur(pSource, pDestination, mKernelSize);
}

void BadPixelReplacement::process8Bit(cv::Mat &pSource, cv::Mat &pDestination)
{
    if(mBadPixelContainer.size() == 0){
        qDebug() << "BadPixelReplacement::process8Bit mBadPixelContainer size == 0 and applied median blur";
        processMedianBlur(pSource, pDestination);
        return;
    }

    QVectorIterator<QPair<int, int>> tIter(mBadPixelContainer);

    pSource.copyTo(pDestination);

    uchar* tData = pDestination.data;

    uint32_t tAvarageOfPixel = 0;
    uint8_t tCountOfPixel = 0;

    while (tIter.hasNext()){
        QPair<int, int> tBadPixel = tIter.next();
        int fromX = std::max(0, tBadPixel.first - mKernelSize);
        int toX = std::min(pDestination.rows, tBadPixel.first + mKernelSize);

        int fromY = std::max(0, tBadPixel.second - mKernelSize);
        int toY = std::min(pDestination.cols, tBadPixel.second + mKernelSize );

        tAvarageOfPixel = 0;
        tCountOfPixel= 0;

        uint8_t tPixelData = tData[ tBadPixel.first * pDestination.cols + tBadPixel.second];

        for( int x = fromX; x < toX; ++x){
            for( int y = fromY; y < toY; ++y){
                if((tData[ x * pDestination.cols + y] != tPixelData) && (tData[ x * pDestination.cols + y] != 0)){
                    ++tCountOfPixel;
                    tAvarageOfPixel += tData[ x * pDestination.cols + y];
                }
            }
        }
        if(tAvarageOfPixel != 0){
            tData[tBadPixel.first * pDestination.cols + tBadPixel.second] = tAvarageOfPixel / tCountOfPixel;
        }

    }
}

void BadPixelReplacement::process16Bit(cv::Mat &pSource, cv::Mat &pDestination)
{
    if(mBadPixelContainer.size() == 0){
        qDebug() << "BadPixelReplacement::process16Bit mBadPixelContainer size == 0 and applied median blur";
        processMedianBlur(pSource, pDestination);
        return;
    }

    QVectorIterator<QPair<int, int>> tIter(mBadPixelContainer);

    pSource.copyTo(pDestination);

    ushort* tData = (ushort*)pDestination.data;

    QPair<int, int> tBadPixel;
    uint32_t tAvarageOfPixel = 0;
    uint8_t tCountOfPixel = 0;

    while (tIter.hasNext()){
        tBadPixel = tIter.next();
        int fromX = std::max(0, tBadPixel.first - mKernelSize);
        int toX = std::min(pDestination.rows, tBadPixel.first + mKernelSize);

        int fromY = std::max(0, tBadPixel.second - mKernelSize);
        int toY = std::min(pDestination.cols, tBadPixel.second + mKernelSize );

        tAvarageOfPixel = 0;
        tCountOfPixel= 0;

        ushort tBadPixelData = tData[ tBadPixel.first * pDestination.cols + tBadPixel.second];

        for( int x = fromX; x < toX; ++x){
            for( int y = fromY; y < toY; ++y){
                ushort tPixelValue = tData[ x * pDestination.cols + y];
                if((tPixelValue != tBadPixelData) && (tPixelValue != 0)){
                    ++tCountOfPixel;
                    tAvarageOfPixel += tPixelValue;
                }
            }
        }
        if(tAvarageOfPixel != 0){
            tData[tBadPixel.first * pDestination.cols + tBadPixel.second] = tAvarageOfPixel / tCountOfPixel;
        }
    }
}

void BadPixelReplacement::processVec(cv::Mat &pSource, cv::Mat &pDestination)
{
    if(mBadPixelContainer.size() == 0){
        qDebug() << "BadPixelReplacement::applyCustom mBadPixelContainer size == 0 and applied median blur";
        processMedianBlur(pSource, pDestination);
        return;
    }

    QVectorIterator<QPair<int, int>> tIter(mBadPixelContainer);

    pSource.copyTo(pDestination);

    ushort* tData = (ushort*)pDestination.data;

    while (tIter.hasNext()){
        QPair<int, int> tBadPixel = tIter.next();
        int fromX = std::max(0, tBadPixel.first - mKernelSize);
        int toX = std::min(pDestination.rows, tBadPixel.first + mKernelSize);

        std::vector<int> tPixelsAroundBadPixel;

        for( int x = fromX; x < toX; x++){
            int fromY = std::max(0, tBadPixel.second - mKernelSize);
            int toY = std::min(pDestination.cols, tBadPixel.second + mKernelSize );
            for( int y = fromY; y < toY; y++){
                tPixelsAroundBadPixel.push_back(tData[ x * pDestination.cols + y]);
            }
        }
        std::sort(tPixelsAroundBadPixel.begin(), tPixelsAroundBadPixel.end());
        if(tPixelsAroundBadPixel.size() > 0){
            tData[tBadPixel.first * pDestination.cols + tBadPixel.second] = tPixelsAroundBadPixel.at(tPixelsAroundBadPixel.size() / 2);
        }
    }
}

void BadPixelReplacement::applyBpr(ushort *pData, int pRows, int pCols){

    QVectorIterator<QPair<int, int>> tIter(mBadPixelContainer);

    QPair<int, int> tBadPixel;
    uint32_t tAvarageOfPixel = 0;
    uint8_t tCountOfPixel = 0;

    while (tIter.hasNext()){
        tBadPixel = tIter.next();
        int fromX = std::max(0, tBadPixel.first - mKernelSize);
        int toX = std::min(pRows, tBadPixel.first + mKernelSize);

        int fromY = std::max(0, tBadPixel.second - mKernelSize);
        int toY = std::min(pCols, tBadPixel.second + mKernelSize );

        tAvarageOfPixel = 0;
        tCountOfPixel= 0;

        ushort tBadPixelData = pData[ tBadPixel.first * pCols + tBadPixel.second];

        for( int x = fromX; x < toX; ++x){
            for( int y = fromY; y < toY; ++y){
                ushort tPixelValue = pData[ x * pCols + y];
                if((tPixelValue != tBadPixelData) && (tPixelValue != 0)){
                    ++tCountOfPixel;
                    tAvarageOfPixel += tPixelValue;
                }
            }
        }
        if((tCountOfPixel != 0) ){
            pData[tBadPixel.first * pCols + tBadPixel.second] = tAvarageOfPixel / tCountOfPixel;
        }
    }
}
