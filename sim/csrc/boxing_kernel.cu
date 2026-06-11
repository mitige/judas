// Judas — simulateur boxing Minecraft 1.8.9, kernels CUDA + binding PyTorch.
//
// 1 thread = 1 match. Toute la logique vit dans boxing_core.h (partagé avec
// le harnais CPU tools/cpu_check.cpp, testé tick par tick contre sim_ref).

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "boxing_core.h"

// -------------------------------------------------------------------- kernels
__global__ void k_reset(StatePtrs S, SimParams pr, float *obs, int n_envs,
                        unsigned long long seed) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_envs) return;
    reset_one(S, pr, obs, n, seed);
}

__global__ void k_tick(StatePtrs S, SimParams pr, const float *actions,
                       float *obs, float *reward, unsigned char *done,
                       int *winner, int n_envs) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_envs) return;
    tick_one(S, pr, actions, obs, reward, done, winner, n);
}

// ------------------------------------------------------------------ launchers
// lancés sur le stream courant PyTorch (pas le stream legacy) : reste ordonné
// avec les ops torch même sous torch.cuda.Stream / AMP
void judas_reset_cuda(StatePtrs S, SimParams pr, float *obs, int n_envs,
                      unsigned long long seed) {
    int threads = 128, blocks = (n_envs + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    k_reset<<<blocks, threads, 0, stream>>>(S, pr, obs, n_envs, seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void judas_tick_cuda(StatePtrs S, SimParams pr, const float *actions,
                     float *obs, float *reward, unsigned char *done,
                     int *winner, int n_envs) {
    int threads = 128, blocks = (n_envs + threads - 1) / threads;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    k_tick<<<blocks, threads, 0, stream>>>(S, pr, actions, obs, reward, done,
                                           winner, n_envs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
