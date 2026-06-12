// Harnais CPU du kernel Judas : exécute boxing_core.h (la logique EXACTE du
// kernel CUDA) en C++ pur, pour vérification contre sim_ref sans GPU.
//
//   g++ -O2 -I sim/csrc -o /tmp/judas_cpu_check tools/cpu_check.cpp
//   /tmp/judas_cpu_check <n_envs> <n_ticks> <actions.bin> <out.bin> <params.txt>
//
// actions.bin : float32 [n_ticks, n_envs, 2, 7]
// params.txt  : 26 floats (SimConfig.as_floats), un par ligne
// out.bin     : par tick -> obs float32 [n_envs,2,48], reward float32 [n_envs,2],
//               done uint8 [n_envs], winner int32 [n_envs]
//               (précédé des obs de reset : float32 [n_envs,2,48])

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "boxing_core.h"

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s n_envs n_ticks actions.bin out.bin params.txt\n",
                argv[0]);
        return 2;
    }
    int n_envs = atoi(argv[1]);
    int n_ticks = atoi(argv[2]);

    FILE *pf = fopen(argv[5], "r");
    if (!pf) { fprintf(stderr, "params introuvables\n"); return 2; }
    double pv[26];
    for (int i = 0; i < 26; ++i)
        if (fscanf(pf, "%lf", &pv[i]) != 1) { fprintf(stderr, "params invalides\n"); return 2; }
    fclose(pf);
    SimParams pr;
    pr.arena_x = (float)pv[0]; pr.arena_z = (float)pv[1];
    pr.target_hits = (float)pv[2]; pr.max_ticks = (float)pv[3]; pr.amp = (float)pv[4];
    pr.cps_min = (float)pv[5]; pr.cps_max = (float)pv[6];
    pr.rot_min = (float)pv[7]; pr.rot_max = (float)pv[8];
    pr.delay_min = (float)pv[9]; pr.delay_max = (float)pv[10];
    pr.jitter = (float)pv[11];
    pr.r_hit = (float)pv[12]; pr.r_hurt = (float)pv[13];
    pr.r_win = (float)pv[14]; pr.r_dist = (float)pv[15];
    pr.randomize = (float)pv[16];
    pr.spawn_gap = (float)pv[17];
    pr.kb_h = (float)pv[18];
    pr.kb_v = (float)pv[19];
    pr.kb_idle = (float)pv[20];
    pr.r_combo = (float)pv[21];
    pr.combo_window = (float)pv[22];
    pr.combo_cap = (float)pv[23];
    pr.smooth_min = (float)pv[24];
    pr.smooth_max = (float)pv[25];

    // état (mêmes layouts que les tenseurs du wrapper Python)
    std::vector<jreal> pos((size_t)n_envs * 2 * 10, (jreal)0.0);
    std::vector<int> ints((size_t)n_envs * 2 * 10, 0);
    std::vector<float> human((size_t)n_envs * 2 * 3, 0.0f);
    std::vector<int> tick(n_envs, 0);
    std::vector<float> queue((size_t)n_envs * 2 * MAX_DELAY * ACT_DIM, 0.0f);
    std::vector<float> last_act((size_t)n_envs * 2 * ACT_DIM, 0.0f);
    std::vector<unsigned long long> rng(n_envs, 0ULL);

    StatePtrs S;
    S.pos = pos.data(); S.ints = ints.data(); S.human = human.data();
    S.tick = tick.data(); S.queue = queue.data(); S.last_act = last_act.data();
    S.rng = rng.data();

    std::vector<float> obs((size_t)n_envs * 2 * OBS_DIM);
    std::vector<float> reward((size_t)n_envs * 2);
    std::vector<unsigned char> done(n_envs);
    std::vector<int> winner(n_envs);
    std::vector<float> actions((size_t)n_envs * 2 * ACT_DIM);

    FILE *fa = fopen(argv[3], "rb");
    FILE *fo = fopen(argv[4], "wb");
    if (!fa || !fo) { fprintf(stderr, "fichiers actions/out invalides\n"); return 2; }

    for (int n = 0; n < n_envs; ++n) reset_one(S, pr, obs.data(), n, 0ULL);
    fwrite(obs.data(), sizeof(float), obs.size(), fo);

    for (int t = 0; t < n_ticks; ++t) {
        if (fread(actions.data(), sizeof(float), actions.size(), fa)
                != actions.size()) {
            fprintf(stderr, "actions tronquées au tick %d\n", t);
            return 2;
        }
        for (int n = 0; n < n_envs; ++n)
            tick_one(S, pr, actions.data(), obs.data(), reward.data(),
                     done.data(), winner.data(), n);
        fwrite(obs.data(), sizeof(float), obs.size(), fo);
        fwrite(reward.data(), sizeof(float), reward.size(), fo);
        fwrite(done.data(), 1, done.size(), fo);
        fwrite(winner.data(), sizeof(int), winner.size(), fo);
    }
    fclose(fa);
    fclose(fo);
    return 0;
}
