#ifndef NONUNIFORMCORRECTION_H
#define NONUNIFORMCORRECTION_H

#include <QString>
#include <QVector>
#include <opencv2/opencv.hpp>

class NonUniformCorrection
{
public:
    NonUniformCorrection(QString pFramesPath);


public:
    void calculte2PointNuc();
    void fillGainOffsetMatrix();

    void applyNuc(ushort *pFrame);

    static const int mFrameCols = 640;
    static const int mFrameRows = 480;

private:

    void calculateMean(QVector<cv::Mat> &tContainer, float pMean[][mFrameRows]);

    void calcDiff();

    void calcMean(float pMean[][mFrameRows], float &pMeanValue);

    void calcGain(float pHotMean, float pColdMean);

    void calcOffset(float pColdMean);


    template<typename T, int cols, int rows>
    std::ostream &writemap(std::ostream& os, T (&map)[cols][rows])
    {
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                os << map[col][row]<<",";
            }
            os<<"\n";
        }
        return os;
    }

    bool readGainOffsetMatrix(float pMatris[][mFrameRows], QString pPath);


private:
    QString mFramesPath;


    QVector<cv::Mat> mColdFrame;
    QVector<cv::Mat> mHotFrame;

    float mHotMean          [mFrameCols][mFrameRows] = {{0.0}};
    float mColdMean         [mFrameCols][mFrameRows] = {{0.0}};
    float mDiffHotCold      [mFrameCols][mFrameRows] = {{0.0}};


    float mGain             [mFrameCols][mFrameRows] = {{0.0}};
    float mOffset           [mFrameCols][mFrameRows] = {{0.0}};
};

#endif // NONUNIFORMCORRECTION_H
