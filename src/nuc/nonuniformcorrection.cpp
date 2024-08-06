#include "nonuniformcorrection.h"

#include <QFile>
#include <qtextstream.h>
#include <fstream>

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "src/Common.h"

NonUniformCorrection::NonUniformCorrection(QString pFramesPath)
    : mFramesPath(pFramesPath)
{

}

void NonUniformCorrection::calculte2PointNuc()
{

    Common::fillFrameContainer(mHotFrame,   mFramesPath + "hot/", 0, 50);
    Common::fillFrameContainer(mColdFrame,  mFramesPath + "cold/", 0, 50);

    calculateMean(mHotFrame, mHotMean);
    calculateMean(mColdFrame, mColdMean);

    calcDiff();

    float tMeanHotMean = 0.0;
    calcMean(mHotMean, tMeanHotMean);

    float tMeanColdMean = 0.0;
    calcMean(mColdMean, tMeanColdMean);

    calcGain(tMeanHotMean, tMeanColdMean);

    calcOffset(tMeanColdMean);


    std::remove(QString(mFramesPath + "gain_matris.csv" ).toStdString().c_str());
    std::remove(QString(mFramesPath + "ofset_matris.csv").toStdString().c_str());


    std::fstream tGainOf(mFramesPath.toStdString() + "gain_matris.csv", std::ios::out | std::ios::app);

    if (tGainOf.is_open())
    {
        writemap(tGainOf, mGain);
        tGainOf.close();
    }

    std::fstream tOffsetOf(mFramesPath.toStdString() + "ofset_matris.csv", std::ios::out | std::ios::app);

    if (tOffsetOf.is_open())
    {
        writemap(tOffsetOf, mOffset);
        tOffsetOf.close();
    }
}

void NonUniformCorrection::fillGainOffsetMatrix()
{
        readGainOffsetMatrix(mGain, mFramesPath + "gain_matris.csv");
        readGainOffsetMatrix(mOffset, mFramesPath + "ofset_matris.csv");
}

void NonUniformCorrection::applyNuc(ushort *pFrame){
    for (int col = 0; col < mFrameCols; ++col) {
        for (int row = 0; row < mFrameRows; ++row) {
            pFrame[row * mFrameCols + col] = (ushort)((mGain[col][row] * pFrame[row * mFrameCols + col]) + mOffset[col][row]);
        }
    }

}

void NonUniformCorrection::calculateMean(QVector<cv::Mat> &tContainer, float pMean[][mFrameRows]){

    for (int col = 0; col < mFrameCols; ++col) {
        for (int row = 0; row < mFrameRows; ++row) {
            long long tMeanVal = 0;
            for (int i = 0; i < tContainer.size(); ++i) {
                tMeanVal += reinterpret_cast<ushort*>(tContainer.at(i).data)[row * mFrameCols + col];
            }
            pMean[col][row] = (float)tMeanVal / tContainer.size();
        }
    }
}

void NonUniformCorrection::calcDiff(){
    for (int col = 0; col < mFrameCols; ++col) {
        for (int row = 0; row < mFrameRows; ++row) {
            mDiffHotCold[col][row] = mHotMean[col][row] - mColdMean[col][row] <= 0 ? 0 : mHotMean[col][row] - mColdMean[col][row];
        }
    }
}

void NonUniformCorrection::calcMean(float pMean[][mFrameRows], float &pMeanValue){

    long double tTemp = 0.0;
    for (int col = 0; col < mFrameCols; ++col) {
        for (int row = 0; row < mFrameRows; ++row) {
            tTemp += pMean[col][row];
        }
    }

    pMeanValue = (float)tTemp / (mFrameCols * mFrameRows);
}

void NonUniformCorrection::calcGain(float pHotMean, float pColdMean){
    for (int col = 0; col < mFrameCols; ++col) {
        for (int row = 0; row < mFrameRows; ++row) {
            mGain[col][row] = (pHotMean - pColdMean) / mDiffHotCold[col][row];
        }
    }
}

void NonUniformCorrection::calcOffset(float pColdMean){
    for (int col = 0; col < mFrameCols; ++col) {
        for (int row = 0; row < mFrameRows; ++row) {
            mOffset[col][row] = pColdMean - (mGain[col][row] * mColdMean[col][row]);
        }
    }
}

bool NonUniformCorrection::readGainOffsetMatrix(float pMatris[][mFrameRows], QString pPath){
    QFile file(pPath);
    if(!file.exists()){
        return false;
    }

    if(!file.open(QIODevice::ReadOnly)){
        return false;
    }

    int i = 0;
    int j = 0;
    QTextStream in(&file);
    while(!in.atEnd()) {
        j = 0;
        QString line = in.readLine();
        QStringList  fields = line.split(",");
        for(const auto &tField: fields){
            pMatris[i][j] = tField.toFloat();
            ++j;
        }
        ++i;
    }
    file.close();
    return true;
}
