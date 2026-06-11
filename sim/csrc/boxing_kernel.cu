// Judas — simulateur boxing Minecraft 1.8.9, kernels CUDA + binding PyTorch.
//
// 1 thread = 1 match. Toute la logique vit dans boxing_core.h (partagé avec
// le harnais CPU tools/cpu_check.cpp, testé tick par tick contre sim_ref).

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

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

// ------------------------------------------------------------------- bindings
static SimParams params_from_vec(const std::vector<double> &v) {
    TORCH_CHECK(v.size() == 21, "SimParams: 21 valeurs attendues");
    SimParams p;
    p.arena_x = (float)v[0]; p.arena_z = (float)v[1];
    p.target_hits = (float)v[2]; p.max_ticks = (float)v[3]; p.amp = (float)v[4];
    p.cps_min = (float)v[5]; p.cps_max = (float)v[6];
    p.rot_min = (float)v[7]; p.rot_max = (float)v[8];
    p.delay_min = (float)v[9]; p.delay_max = (float)v[10];
    p.jitter = (float)v[11];
    p.r_hit = (float)v[12]; p.r_hurt = (float)v[13];
    p.r_win = (float)v[14]; p.r_dist = (float)v[15];
    p.randomize = (float)v[16];
    p.spawn_gap = (float)v[17];
    p.kb_h = (float)v[18];
    p.kb_v = (float)v[19];
    p.kb_idle = (float)v[20];
    return p;
}

static StatePtrs ptrs_from_tensors(torch::Tensor pos, torch::Tensor ints,
                                   torch::Tensor human, torch::Tensor tick,
                                   torch::Tensor queue, torch::Tensor last_act,
                                   torch::Tensor rng) {
    TORCH_CHECK(pos.scalar_type() == (sizeof(jreal) == 8 ? torch::kFloat64
                                                         : torch::kFloat32),
                "dtype de pos incompatible avec la precision du build");
    StatePtrs S;
    S.pos = pos.data_ptr<jreal>();
    S.ints = ints.data_ptr<int>();
    S.human = human.data_ptr<float>();
    S.tick = tick.data_ptr<int>();
    S.queue = queue.data_ptr<float>();
    S.last_act = last_act.data_ptr<float>();
    S.rng = reinterpret_cast<unsigned long long *>(rng.data_ptr<int64_t>());
    return S;
}

void judas_reset(torch::Tensor pos, torch::Tensor ints, torch::Tensor human,
                 torch::Tensor tick, torch::Tensor queue, torch::Tensor last_act,
                 torch::Tensor rng, torch::Tensor obs,
                 std::vector<double> params, int64_t seed) {
    int n = (int)pos.size(0);
    StatePtrs S = ptrs_from_tensors(pos, ints, human, tick, queue, last_act, rng);
    SimParams pr = params_from_vec(params);
    int threads = 128, blocks = (n + threads - 1) / threads;
    k_reset<<<blocks, threads>>>(S, pr, obs.data_ptr<float>(), n,
                                 (unsigned long long)seed);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void judas_tick(torch::Tensor pos, torch::Tensor ints, torch::Tensor human,
                torch::Tensor tick, torch::Tensor queue, torch::Tensor last_act,
                torch::Tensor rng, torch::Tensor actions, torch::Tensor obs,
                torch::Tensor reward, torch::Tensor done, torch::Tensor winner,
                std::vector<double> params) {
    int n = (int)pos.size(0);
    StatePtrs S = ptrs_from_tensors(pos, ints, human, tick, queue, last_act, rng);
    SimParams pr = params_from_vec(params);
    int threads = 128, blocks = (n + threads - 1) / threads;
    k_tick<<<blocks, threads>>>(S, pr, actions.data_ptr<float>(),
                                obs.data_ptr<float>(), reward.data_ptr<float>(),
                                done.data_ptr<unsigned char>(),
                                winner.data_ptr<int>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reset", &judas_reset, "Judas reset kernel");
    m.def("tick", &judas_tick, "Judas tick kernel");
}
