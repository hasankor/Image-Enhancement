#include "dehaze.h"
#include <iostream>
#include <chrono>

#include "cuda.h"
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "darkchannel.h"
#include "dehaze_kernel.h"
#include "boxfilter.h"
#include "../Utils.h"


void dehazeGPU(unsigned char *pSrcData, unsigned char * pOutData, int channel, int width, int height){
    float atmos_correct = 0.0;
    float eps = 1e-05f;
    const int tRadius = 7;

    unsigned char   *d_ori;
    unsigned char	*d_dehaze;

    float			*d_fog;
    float           *d_min_img;
    float			*d_win_dark;
    int				*d_index;
    float			*d_im_dark;
    float           *d_dark_mat;
    float			*d_im;
    float			*d_tDown;
    float			*d_foggy_gray;
    float			*d_atmosLight;
    float           *d_atmos;

    float			*d_mean_I;
    float			*d_mean_p;
    float           *d_mean_Ip;
    float           *d_cov_Ip;
    float           *d_mean_II;
    float           *d_var_I;
    float           *d_a;
    float           *d_b;
    float           *d_mean_a;
    float           *d_mean_b;
    float			*d_box_temp;
    float			*d_filtered;
    float			*d_t;
    float			*d_temp;

    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_ori, height*width*channel*sizeof(unsigned char)));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_fog, sizeof(float)*width*height*channel));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_dehaze, sizeof(unsigned char)*width*height*channel));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_min_img, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_win_dark, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_im_dark, sizeof(float)*width*height));

    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_dark_mat, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_index, sizeof(int)*width*height));//2048 is enough
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_im, sizeof(float)*width*height*channel));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_temp, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_atmosLight, sizeof(float) * 3));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_atmos, sizeof(float) * 3));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_filtered, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_t, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_foggy_gray, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_tDown, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_mean_I, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_mean_p, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_mean_Ip, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_cov_Ip, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_mean_II, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_var_I, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_a, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_b, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_mean_a, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_mean_b, sizeof(float)*width*height));
    GPU_HANDLE_ERROR(cudaMalloc((void **)&d_box_temp, sizeof(float)*width*height));

    GPU_HANDLE_ERROR(cudaMemcpy(d_ori, pSrcData, sizeof(unsigned char)*width*height*channel, cudaMemcpyHostToDevice));

    float_fog_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_ori, d_fog, d_foggy_gray, width, height, channel);


    minfilter(d_fog, d_min_img, d_win_dark, d_temp, width, height, channel, tRadius);
    MaxReductionkernel << <(height * width ) / 1024 +1, 512 >> >(d_win_dark, d_im_dark, width, height, d_index);//Find 2048 max value into d_index
    MaxReductionkernelTwo << < height * width / 1024 / 512 + 1, 256 >> >(d_im_dark, d_win_dark, d_foggy_gray, width, height, d_index);
    atomsLight_kernel << <1, 1 >> >(d_fog, d_win_dark, d_index, d_im, d_atmosLight, d_atmos, width, height, channel, tRadius);
    atomsLight_kernel_divide << <(height*width + BLOCKSIZE-1) / BLOCKSIZE, BLOCKSIZE >> >(d_fog, d_im, d_atmosLight, width, height, channel);


    minfilter(d_im, d_min_img, d_dark_mat, d_temp, width, height, channel, tRadius);
    t_initial_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_dark_mat, d_tDown, width, height, channel);

    //Guided Filter

    boxfilter(d_foggy_gray, d_mean_I, d_temp, height, width, tRadius * 5);
    boxfilter(d_tDown, d_mean_p, d_temp, height, width, tRadius * 5);
    matrix_dot_multiple << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_foggy_gray, d_tDown, d_box_temp, height, width);

    boxfilter(d_box_temp, d_mean_Ip, d_temp, height, width, tRadius * 5);
    matrix_dot_multiple << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_foggy_gray, d_foggy_gray, d_box_temp, height, width);

    boxfilter(d_box_temp, d_mean_II, d_temp, height, width, tRadius * 5);
    matrix_cal_a << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_mean_I, d_mean_p, d_mean_II, d_mean_Ip, d_a, eps, height, width);

    matrix_cal_b << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_mean_I, d_mean_p, d_mean_II, d_mean_Ip, d_b, eps, height, width);

    boxfilter(d_a, d_mean_a, d_temp, height, width, tRadius * 5);

    boxfilter(d_b, d_mean_b, d_temp, height, width, tRadius * 5);
    matrix_cal_q << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_mean_a, d_mean_b, d_foggy_gray, d_filtered, width, height);


    clear_kernel << <(height*width + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >> >(d_fog, d_dehaze, d_atmosLight, d_filtered, width, height, channel, tRadius, atmos_correct);



    GPU_HANDLE_ERROR(cudaPeekAtLastError());

    // Wait for all threads to finish
    GPU_HANDLE_ERROR(cudaDeviceSynchronize());

    GPU_HANDLE_ERROR(cudaMemcpy(pOutData, d_dehaze, width*height*channel * sizeof(unsigned char), cudaMemcpyDeviceToHost));


    GPU_HANDLE_ERROR(cudaFree(d_ori));
    GPU_HANDLE_ERROR(cudaFree(d_fog));
    GPU_HANDLE_ERROR(cudaFree(d_dehaze));
    GPU_HANDLE_ERROR(cudaFree(d_min_img));
    GPU_HANDLE_ERROR(cudaFree(d_win_dark));
    GPU_HANDLE_ERROR(cudaFree(d_im_dark));
    GPU_HANDLE_ERROR(cudaFree(d_index));
    GPU_HANDLE_ERROR(cudaFree(d_dark_mat));
    GPU_HANDLE_ERROR(cudaFree(d_im));
    GPU_HANDLE_ERROR(cudaFree(d_tDown));
    GPU_HANDLE_ERROR(cudaFree(d_foggy_gray));
    GPU_HANDLE_ERROR(cudaFree(d_atmos));
    GPU_HANDLE_ERROR(cudaFree(d_atmosLight));
    GPU_HANDLE_ERROR(cudaFree(d_filtered));
    GPU_HANDLE_ERROR(cudaFree(d_t));
    GPU_HANDLE_ERROR(cudaFree(d_mean_I));
    GPU_HANDLE_ERROR(cudaFree(d_mean_p));
    GPU_HANDLE_ERROR(cudaFree(d_mean_Ip));
    GPU_HANDLE_ERROR(cudaFree(d_cov_Ip));
    GPU_HANDLE_ERROR(cudaFree(d_mean_II));
    GPU_HANDLE_ERROR(cudaFree(d_var_I));
    GPU_HANDLE_ERROR(cudaFree(d_a));
    GPU_HANDLE_ERROR(cudaFree(d_b));
    GPU_HANDLE_ERROR(cudaFree(d_mean_a));
    GPU_HANDLE_ERROR(cudaFree(d_mean_b));
    GPU_HANDLE_ERROR(cudaFree(d_box_temp));
    GPU_HANDLE_ERROR(cudaFree(d_temp));
}
