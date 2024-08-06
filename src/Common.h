#ifndef COMMON_H
#define COMMON_H

#include <QString>
#include <QVector>
#include <opencv2/opencv.hpp>

class Common
{

public:

    static void fillFrameContainer(QVector<cv::Mat> &tContainer, QString pPath, int pStart = 0, int pCount = 50){
        for (uint i = pStart; i <= pCount; ++i) {
            cv::Mat tFrame = cv::imread(QString(pPath + QString::number(i) + ".tiff").toStdString().c_str(), cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
            tContainer.push_back(tFrame);
        }
    }


    static void show8BitImage(ushort *pData, std::string pName, int rows = 480, int cols = 640){
        uchar *tData = new uchar[rows * cols];
        for(uint row = 0; row < rows; ++row){
            for(uint col = 0; col < cols; ++col){
                tData[row * cols + col] =  pData[row * cols + col] * 255 / 16383;
            }
        }

        cv::Mat tFrame(rows, cols, CV_8UC1, tData);
        cv::imshow(pName, tFrame);
        tFrame.release();
        delete [] tData;
    }

private:
    Common() {}
};



#endif // COMMON_H
