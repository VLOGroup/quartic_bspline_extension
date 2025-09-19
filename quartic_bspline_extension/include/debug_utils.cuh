#pragma once

#include <cstdio>
#include <cuda_runtime.h>

/* 
    NOTE
    ----
        > To disable debug messages set 
            #define CUDA_DEBUG_FLAG 0
        > To enable debug messages set
            #define CUDA_DEBUG_FLAG 1
*/
#define CUDA_DEBUG_FLAG 0

inline void cuda_debug_func(cudaError_t err,
                            const char* file,
                            int line){
    #ifdef CUDA_DEBUG_FLAG
        if (cudaSuccess != err) {
            fprintf(stderr,
                    "CUDA error at %s:%d: %s\n",
                    file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    #endif
}

#define CUDA_DEBUG_FUNC(call) cuda_debug_func((call), __FILE__, __LINE__)