#ifndef DEFOGGING_H
#define DEFOGGING_H

#include <opencv2/core.hpp>
#include <iostream>

class Defogger
{
public:
    Defogger();

    void defog(cv::Mat pSource, cv::Mat& pOutput, int pRectSize = 15, double pOmega = 0.95, double pNumt = 0.1);

private:

    /*!
     * \brief darkChannel
     * \param pSource
     * \param pSize
     * \return
     * Find a dark channel
     * \note: The size of the window is a key parameter for the result. The larger the window, the greater the probability of containing dark channels,
     *  the darker the dark channels, and the less obvious the effect of defogging. The general window size is 11-51 Between, that is,
     *  the radius is between 5-25.
     */
    cv::Mat darkChannel(cv::Mat pSource, int pSize);

    /*!
     * \brief atmLight
     * \param pSource
     * \param pDark
     * \return
     * \note: Find the global atmospheric light value A
     */
    void tAtmLight(cv::Mat pSource, cv::Mat pDark, float pOutA[3]);

    /*!
     * \brief transmissionEsticv::Mate
     * \param pSource
     * \param pOutA
     * \param pSize
     * \param pOmega
     * \return
     * \note: Calculate and calculate the estimated value of transmittance
     * The omega in has obvious meaning, the smaller the value, the less obvious the defogging effect
     */
    cv::Mat transmissionEstimate(cv::Mat pSource, float pOutA[3], int pSize, float pOmega);

    /*!
     * \brief guidedFilter
     * \param pSource
     * \param pTransmissionEsticv::Mate
     * \param pR
     * \param pEps
     * \return
     * \note:Guided filtering
     */
    cv::Mat guidedfilter(cv::Mat pSource, cv::Mat pTransmissionEstimated, int pR, float pEps);

    /*!
     * \brief transmissionRefine
     * \param pSource
     * \param pTransmissionEsticv::Mate
     * \return
     * \note:Calculation of transmittance by guided filtering
     */
    cv::Mat transmissionRefine(cv::Mat pSource, cv::Mat pTransmissionEstimated);

    /*!
     * \brief recover
     * \param pSource
     * \param pTransmission
     * \param pOutA
     * \param pTx
     * \return
     * \note: Image defogging
     */
    cv::Mat recover(cv::Mat pSource, cv::Mat pTransmissionRefined, float pOutA[3], float pTx);

private:
    template<typename T>
    std::vector<int> argsort(const std::vector<T>& pArray)
    {
        const int tArrayLen(pArray.size());
        std::vector<int> tArrayIndex(tArrayLen, 0);
        for (int i = 0; i < tArrayLen; ++i)
            tArrayIndex[i] = i;

        sort(tArrayIndex.begin(), tArrayIndex.end(), [&pArray](int tPos1, int tPos2) {
            return (pArray[tPos1] < pArray[tPos2]);
        });

        return tArrayIndex;
    }

private:
    std::vector<std::vector<int>> matrixReshape(std::vector<std::vector<int>>& nums, int r, int c) {
         std::vector<int> temp;
         std::vector<std::vector<int> > res(r, std::vector<int>(c));
         int count = 0;
         for (int i = 0; i < static_cast<int>(nums.size()); i++) {
            for (int j = 0; j < static_cast<int>(nums[0].size()); j++) {
               temp.push_back(nums[i][j]);
            }
         }
         if (r * c != static_cast<int>(nums.size()) * static_cast<int>(nums[0].size()))
            return nums;
         for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
               res[i][j] = temp[count++];
            }
         }
         return res;
      }
};

#endif // DEFOGGING_H
