#include "cpustretcher.h"
#include "iostream"

CpuStretcher::CpuStretcher()
{

}

void CpuStretcher::stretch14Bit(ushort *pData, int pRows, int pCols){

    QList<ushort> tContiner;
    for(int row = 0; row < pRows; ++row){
        for(int col = 0; col < pCols; ++col){
            ushort tData = pData[row * pCols + col];
            if(tData != 0 && tData != 16383){
                tContiner.append(tData);
            }
        }
    }

    std::sort(tContiner.begin(), tContiner.end());

    ushort min = tContiner[tContiner.size()/2] * 0.25, max = tContiner[tContiner.size()/2] * 0.35;

    float diff = 16383.0f / (float)(max - min);
    for(int row = 0; row < pRows; ++row){
        for(int col = 0; col < pCols; ++col){
            pData[row * pCols + col] = (pData[row * pCols + col] - min) * diff;
        }
    }
}
