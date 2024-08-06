#ifndef CPUSTRETCHER_H
#define CPUSTRETCHER_H

#include <QList>

class CpuStretcher
{
public:
    CpuStretcher();

    void stretch14Bit(ushort *pData, int pRows = 480, int pCols = 640);
};

#endif // CPUSTRETCHER_H
