#ifndef DETECTORDECODER_H
#define DETECTORDECODER_H

#include <gst/gst.h>
#include <src/video/decoder/avideoimagedecoder.h>

class Defogger;
class HazeRemove;
class Defogger16bit;
class CudaImageProcessor;
class BadPixelReplacement;

class DetectorDecoder : public AVideoImageDecoder
{
public:
    DetectorDecoder();

    // AVideoImageDecoder interface
private:
    QString createGstPipelineString() override;
    QString getAppSinkName() override;

    // AVideoImageDecoder interface
private:
    bool imageProcess(uchar *pData);

private:
    bool readGainOffsetMatrix();

    // AVideoImageDecoder interface
private:
    QImage::Format getFormat();

private:
    Defogger *mDefogger;
    HazeRemove *mHazeRemove;
    Defogger16bit *mDefogger16bit;
    CudaImageProcessor *mCudaImageProcessor;
    BadPixelReplacement *mBadPixelReplacement;

private:
    int mKernelSize = 3;

    /*!
     * \brief mBadPixelContainer
     * @arg x, y coordinate
     */
    QVector<QPair<int, int>> mBadPixelContainer;

};

#endif // DETECTORDECODER_H
