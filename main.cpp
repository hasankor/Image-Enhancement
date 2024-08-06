#include <QCoreApplication>
#include <src/defogger/defogger.h>
#include <src/hazeremove/hazeremove.h>
#include <src/defogger16bit/defogger16bit.h>
#include <src/badpixelreplacement/badpixelreplacement.h>
#include <cudasrc/imageprocessor/imageprocessor.h>
#include <src/nuc/nonuniformcorrection.h>
#include <src/autobrightnessconstrast/autobrightnessandcontrast.h>
#include <src/stretch/cpustretcher.h>
#include "src/Common.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    QString tPath = "/home/hasankor/Pictures/DenoisingAndEnhacement-main/data/";


    QVector<cv::Mat> tRawFrame;
    Common::fillFrameContainer(tRawFrame, tPath + "normal/", 0, 191);

    Defogger *tDefogger = new Defogger();
    HazeRemove *tHazeRemove = new HazeRemove();
    CpuStretcher *tStretcher = new CpuStretcher();
    Defogger16bit *tDefogger16bit = new Defogger16bit();
    NonUniformCorrection *tNonUniformCorrection = new NonUniformCorrection(tPath);
    AutoBrightnessAndContrast *tAutoBrightnessAndContrast = new AutoBrightnessAndContrast();
    BadPixelReplacement *tBadPixelReplacement = new BadPixelReplacement(tPath + "deadpixel.csv");

    CudaImageProcessor *tCudaImageProcessor = new CudaImageProcessor();

    tNonUniformCorrection->calculte2PointNuc();
    tNonUniformCorrection->fillGainOffsetMatrix();

    int tIndex = 0;
    while (tIndex < tRawFrame.size()) {
        cv::Mat tFrame = tRawFrame[tIndex].clone();

        ushort *tFrameData = reinterpret_cast<ushort*>(tFrame.data);
        Common::show8BitImage(tFrameData, "Raw");

        tNonUniformCorrection->applyNuc(tFrameData);
        Common::show8BitImage(tFrameData, "Nuc");

        tBadPixelReplacement->applyBpr(tFrameData);
        Common::show8BitImage(tFrameData, "Apply Bpr");

        cv::Mat tOutputCuda;
        tCudaImageProcessor->process(tFrame.clone(), tOutputCuda);
        cv::imshow("Cuda Process", tOutputCuda);

        tStretcher->stretch14Bit(tFrameData);
        Common::show8BitImage(tFrameData, "Stretching");

        std::tuple<cv::Mat, float, float> tResult = tAutoBrightnessAndContrast->automaticBrightnessAndContrast14bit(tFrame);
        Common::show8BitImage(tFrameData, "Auto Brightness And Contrast");


        ///////////////16bitDefogger/////////////////////
        cv::Mat tOutput16bitDefog, tFrame16bit = tFrame.clone();
        cv::cvtColor(tFrame16bit, tFrame16bit, cv::COLOR_GRAY2RGB);
        tDefogger16bit->defog(tFrame16bit, tOutput16bitDefog);
        cv::imshow("Haze Remove 16Bit", tOutput16bitDefog);
        //////////////////////////////////


        ///below codes: just work on rgb color image

        ///////////////16bitDefogger/////////////////////
        cv::Mat tOutputDefog, tFrameInput = tFrame.clone();
        cv::cvtColor(tFrameInput, tFrameInput, cv::COLOR_GRAY2RGB);
        tDefogger->defog(tFrameInput, tOutputDefog);
        cv::imshow("Defog", tOutputDefog);
        //////////////////////////////////


        ///////////////16bitDefogger/////////////////////
        cv::Mat tOutputHazeRemove;
        tFrameInput = tFrame.clone();
        cv::cvtColor(tFrameInput, tFrameInput, cv::COLOR_GRAY2RGB);
        tHazeRemove->process(tFrameInput, tOutputHazeRemove);
        cv::imshow("Haze Remove", tOutputHazeRemove);
        //////////////////////////////////

        cv::waitKey(1);
        ++tIndex;
        if(tIndex == tRawFrame.size()){
            tIndex = 0;
        }
        tFrame.release();
    }

    return app.exec();
}

