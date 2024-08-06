#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime_api.h>
#include <stdio.h>

#define MAX_SIZE 3000
#define MAX_VALUE 255.0
#define TILE_DIM 16
#define BLOCK_DIM 1024
#define BLOCKSIZE 256

#define GPU_HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif // UTILS_H
