#include "detectordecoder.h"
#include <opencv2/opencv.hpp>
#include <src/defogger/defogger.h>
#include <src/hazeremove/hazeremove.h>
#include <src/defogger16bit/defogger16bit.h>
#include <src/badpixelreplacement/badpixelreplacement.h>
#include <cudasrc/imageprocessor/imageprocessor.h>
#include <QFile>
#include <QStandardPaths>
#include <chrono>

//static unsigned long long mFrameCount = 0;
//static unsigned long long mGPUDuration = 0;
//static unsigned long long mBPRDuration = 0;
//static unsigned long long mDefogDuration = 0;

float G_OFFSETMATRIX[480][640] = {0};
float G_GAINMATRIX  [480][640] = {0};

DetectorDecoder::DetectorDecoder()
    : AVideoImageDecoder()
{
    mDefogger = new Defogger();
    mHazeRemove = new HazeRemove();
    mDefogger16bit = new Defogger16bit();
    mCudaImageProcessor = new CudaImageProcessor();
    mBadPixelReplacement = new BadPixelReplacement("/home/trvgpuserver/Desktop/Projects/DNTSSAlgorithm/backup/dedector_bpr.csv");

    readGainOffsetMatrix();

    QFile tFile("/home/trvgpuserver/Desktop/Projects/DNTSSAlgorithm/backup/dedector_bpr.csv");
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

QString DetectorDecoder::createGstPipelineString()
{
    static QString tPipeline = QString("udpsrc buffer-size=600000 caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW,"
                                       " sampling=(string)YCbCr-4:2:2, depth=(string)8, width=(string)640, height=(string)480, colorimetry=(string)BT709-2, payload=(int)100\""
                                       " port=50001 ! rtpvrawdepay !   videoconvert ! video/x-raw,format=GRAY16_LE ! queue "
                                       "! appsink name = %1 sync=false drop=true max-buffers=1 enable-last-sample=false").arg(getAppSinkName());

    return tPipeline;
}

QString DetectorDecoder::getAppSinkName()
{
    return QString("detectorDecoderAppSink");
}

void show8BitImage(ushort *pData, std::string pName, uint pRows, uint pCols){
    uchar *tData = new uchar[pRows * pCols];
    for(uint row = 0; row < pRows; ++row){
        for(uint col = 0; col < pCols; ++col){
            tData[row * pCols + col] =  pData[row * pCols + col] * 255 / 16383;
        }
    }

    cv::Mat tFrame(pRows, pCols, CV_8UC1, tData);
    cv::imshow(pName, tFrame);
}

bool DetectorDecoder::imageProcess(uchar *pData)
{
    unsigned short *tFrameData = reinterpret_cast<unsigned short*>(pData);

    show8BitImage(tFrameData, "OpencvFrame", getRealHeight(), getRealWidth());

    //////////////////////////NUC/////////////////////////
    for(int row = 0; row < getRealHeight(); ++row){
        for(int col = 0; col < getRealWidth(); ++col){
            tFrameData[row * getRealWidth() + col] =  (G_GAINMATRIX[row][col] * tFrameData[row * getRealWidth() + col]) + G_OFFSETMATRIX[row][col];
            if(tFrameData[row * getRealWidth() + col] > 16383){
                qDebug() << "out of range:" << tFrameData[row * getRealWidth() + col];
            }
        }
    }
    show8BitImage(tFrameData, "NUC", getRealHeight(), getRealWidth());
    //////////////////////////////////////////////////////


    //////////////////////BPR////////////////////////////////
    QVectorIterator<QPair<int, int>> tIter(mBadPixelContainer);

    QPair<int, int> tBadPixel;
    uint32_t tAvarageOfPixel = 0;
    uint8_t tCountOfPixel = 0;

    while (tIter.hasNext()){
        tBadPixel = tIter.next();
        int fromX = std::max(0, tBadPixel.first - mKernelSize);
        int toX = std::min(getRealHeight(), tBadPixel.first + mKernelSize);

        int fromY = std::max(0, tBadPixel.second - mKernelSize);
        int toY = std::min(getRealWidth(), tBadPixel.second + mKernelSize );

        tAvarageOfPixel = 0;
        tCountOfPixel= 0;

        ushort tBadPixelData = tFrameData[ tBadPixel.first * getRealWidth() + tBadPixel.second];

        for( int x = fromX; x < toX; ++x){
            for( int y = fromY; y < toY; ++y){
                ushort tPixelValue = tFrameData[ x * getRealWidth() + y];
                if((tPixelValue != tBadPixelData) && (tPixelValue != 0)){
                    ++tCountOfPixel;
                    tAvarageOfPixel += tPixelValue;
                }
            }
        }
        if(tAvarageOfPixel != 0){
            tFrameData[tBadPixel.first * getRealWidth() + tBadPixel.second] = tAvarageOfPixel / tCountOfPixel;
        }

    }
    show8BitImage(tFrameData, "BPR", getRealHeight(), getRealWidth());
    //////////////////////////////////////////////////////


    ///////////////Stretching///////////////
    QList<ushort> tContiner;
    for(int row = 0; row < getRealHeight(); ++row){
        for(int col = 0; col < getRealWidth(); ++col){
            ushort tData = tFrameData[row * getRealWidth() + col];
            if(tData != 0 && tData != 16383){
                tContiner.append(tData);
            }
        }
    }

    std::sort(tContiner.begin(), tContiner.end());

    ushort min = tContiner[tContiner.size()/2] * 0.5, max = tContiner[tContiner.size()/2] * 1.5;
//    ushort min = tContiner[10], max = tContiner[tContiner.size() - 10];

    qDebug() << "min:" << min << " max:" << max;
    qDebug() << "mid:" << tContiner[tContiner.size()/2];
    qDebug() << "///////////////////////////////////////";

    auto diff = 16383.0f / (float)(max - min);
    for(int row = 0; row < getRealHeight(); ++row){
        for(int col = 0; col < getRealWidth(); ++col){
            tFrameData[row * getRealWidth() + col] = (tFrameData[row * getRealWidth() + col] - min) * diff;
        }
    }
    show8BitImage(tFrameData, "STRETCH", getRealHeight(), getRealWidth());
    ////////////////////////////////////////


    //    ///////////BPR Process/////////////
    //    auto start = std::chrono::high_resolution_clock::now();
    //    cv::Mat tOutputBPR;
    //    mBadPixelReplacement->process16Bit(tFrame, tOutputBPR);
    //    cv::imshow("BPR", tOutputBPR);
    //    qDebug() << "BPR Duration Of Each Frame MicroSec:" <<  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();

    //    cv::imshow("BPR", tOutputBPR);
    //    /////////////////////////////////////////
    //    ///
    //    /////////////CUDA Process/////////////
    //    cv::Mat tOutputCuda;
    //    start = std::chrono::high_resolution_clock::now();
    //    mCudaImageProcessor->process(tFrame, tOutputCuda);
    //    qDebug() << "GPU Duration Of Each Frame MicroSec:" <<  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
    //    cv::imshow("Cuda Process", tOutputCuda);
    //    /////////////////////////////////////////





    //     ///////////////defogger//////////////////////

    //     ///////////////16bitDefogger/////////////////////
    //     cv::Mat tOutput16bitDefog;
    //     cv::cvtColor(tOutputCuda, tOutputCuda, cv::COLOR_GRAY2RGB);
    //     start = std::chrono::high_resolution_clock::now();
    //     mDefogger16bit->defog(tOutputCuda, tOutput16bitDefog);
    //     qDebug() << "Each Haze 16Bit Defogger Process Time:" << (mDefogDuration += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()) /  mFrameCount;
    // //    cv::imshow("Haze 16Bit Removing", tOutput16bitDefog);
    //     //////////////////////////////////


    return true;
}

bool DetectorDecoder::readGainOffsetMatrix(){
    QFile file("/home/trvgpuserver/Desktop/Projects/DNTSSAlgorithm/backup/gain_and_offset_matrix3.csv");
    int i = 0;
    int j = 0;
    if(file.exists()){
        if(file.open(QIODevice::ReadOnly)){
            QTextStream in(&file);
            QString line = in.readLine();
            while(!in.atEnd()) {

                QStringList  fields = line.split(" ");
                j = 0;
                for(const auto &gainAndOffsetField: fields){
                    QStringList gainOffsetStr = gainAndOffsetField.split(",");
                    if(gainOffsetStr.size() == 2){
                        G_GAINMATRIX[i][j] = gainOffsetStr[0].toFloat();
                        G_OFFSETMATRIX[i][j] = gainOffsetStr[1].toFloat();
                    }

                    j++;
                }
                i++;
                line = in.readLine();
            }
        }
        return true;
    }
    return false;
}

QImage::Format DetectorDecoder::getFormat()
{
    return QImage::Format_Grayscale8;
}


