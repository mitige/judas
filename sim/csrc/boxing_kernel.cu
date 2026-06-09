// Judas — simulateur boxing Minecraft 1.8.9, kernel CUDA fusionné.
//
// 1 thread = 1 match (2 agents simulés séquentiellement, exactement dans
// l'ordre de sim_ref/match.py). Positions/vitesses/rotations en double pour
// l'équivalence bit-près avec la référence Python (tests/test_equivalence.py).
//
// SOURCE DE VÉRITÉ de la physique : sim_ref/ (Python). Toute modification
// ici doit être reflétée là-bas et inversement.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define MAX_DELAY 8
#define ACT_DIM 7
#define OBS_DIM 48

// ------------------------------------------------------------- constantes 1.8.9
#define PLAYER_HALF_WIDTH 0.3
#define PLAYER_HEIGHT 1.8
#define PLAYER_EYE_HEIGHT 1.62
#define INPUT_FACTOR 0.98
#define GROUND_FRICTION 0.546          // 0.6 * 0.91
#define AIR_DRAG_H 0.91
#define MAGIC_GROUND 0.16277136
#define GRAVITY 0.08
#define AIR_DRAG_V 0.98
#define JUMP_MOTION_Y 0.42
#define SPRINT_JUMP_BOOST 0.2
#define JUMP_COOLDOWN_TICKS 10
#define AIR_MOVE_FACTOR 0.02
#define BASE_MOVE_SPEED 0.10000000149011612
#define SPRINT_MODIFIER 0.3
#define SPEED_POTION_PER_LEVEL 0.20000000298023224
#define ATTACK_REACH 3.0
#define COLLISION_BORDER 0.1
#define KNOCKBACK_STRENGTH 0.4
#define KNOCKBACK_Y_CAP 0.4
#define SPRINT_KB_H 0.5
#define SPRINT_KB_Y 0.1
#define ATTACKER_SLOWDOWN 0.6
#define MAX_HURT 20
#define HURT_REHIT 10
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232

struct SimParams {
    float arena_x, arena_z;
    float target_hits, max_ticks, amp;
    float cps_min, cps_max, rot_min, rot_max, delay_min, delay_max;
    float jitter;
    float r_hit, r_hurt, r_win, r_dist;
    float randomize;
};

// État d'un agent en registres
struct P {
    double x, y, z, vx, vy, vz, yaw, pitch;
    int hurt, jt, ccd, hits;
    int og, spr, col;          // on_ground, sprinting, collided_horizontally
    float h_cps, h_rot;        // humanisation
    int h_delay;
};

// ----------------------------------------------------------------------- RNG
__device__ inline double rng_next(unsigned long long &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (double)(s >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
}

__device__ inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline double wrap_degrees(double a) {
    a = fmod(a, 360.0);
    if (a >= 180.0) a -= 360.0;
    if (a < -180.0) a += 360.0;
    return a;
}

// ------------------------------------------------------------------ mouvement
__device__ inline double move_speed(int sprinting, int amp) {
    double s = BASE_MOVE_SPEED;
    if (amp >= 0) s *= 1.0 + SPEED_POTION_PER_LEVEL * (double)(amp + 1);
    if (sprinting) s *= 1.0 + SPRINT_MODIFIER;
    return s;
}

__device__ inline void do_jump(P &p) {
    p.vy = JUMP_MOTION_Y;
    if (p.spr) {
        double yr = p.yaw * DEG2RAD;
        p.vx -= sin(yr) * SPRINT_JUMP_BOOST;
        p.vz += cos(yr) * SPRINT_JUMP_BOOST;
    }
}

__device__ inline void move_flying(P &p, double strafe, double forward, double friction) {
    double f = strafe * strafe + forward * forward;
    if (f >= 1.0e-4) {
        f = sqrt(f);
        if (f < 1.0) f = 1.0;
        f = friction / f;
        strafe *= f; forward *= f;
        double s = sin(p.yaw * DEG2RAD), c = cos(p.yaw * DEG2RAD);
        p.vx += strafe * c - forward * s;
        p.vz += forward * c + strafe * s;
    }
}

__device__ inline void move_entity(P &p, double dx, double dy, double dz,
                                   double ax, double az) {
    double dx0 = dx, dy0 = dy, dz0 = dz;
    double ny = p.y + dy;
    if (ny < 0.0) { ny = 0.0; dy = ny - p.y; }
    p.y = ny;
    double lo = PLAYER_HALF_WIDTH, hx = ax - PLAYER_HALF_WIDTH;
    double nx = p.x + dx;
    if (nx < lo) { nx = lo; dx = nx - p.x; }
    else if (nx > hx) { nx = hx; dx = nx - p.x; }
    p.x = nx;
    double hz = az - PLAYER_HALF_WIDTH;
    double nz = p.z + dz;
    if (nz < lo) { nz = lo; dz = nz - p.z; }
    else if (nz > hz) { nz = hz; dz = nz - p.z; }
    p.z = nz;
    bool cx = dx0 != dx, cy = dy0 != dy, cz = dz0 != dz;
    p.col = (cx || cz) ? 1 : 0;
    p.og = (cy && dy0 < 0.0) ? 1 : 0;
    if (cx) p.vx = 0.0;
    if (cy) p.vy = 0.0;
    if (cz) p.vz = 0.0;
}

__device__ inline void living_update_movement(P &p, double strafe_in, double forward_in,
                                              int jumping, double ax, double az, int amp) {
    double strafe = strafe_in * INPUT_FACTOR;
    double forward = forward_in * INPUT_FACTOR;
    if (p.jt > 0) p.jt -= 1;
    if (jumping) {
        if (p.og && p.jt == 0) { do_jump(p); p.jt = JUMP_COOLDOWN_TICKS; }
    } else p.jt = 0;

    double f4, accel;
    if (p.og) {
        f4 = GROUND_FRICTION;
        accel = move_speed(p.spr, amp) * (MAGIC_GROUND / (f4 * f4 * f4));
    } else {
        f4 = AIR_DRAG_H;
        accel = AIR_MOVE_FACTOR * (p.spr ? 1.3 : 1.0);
    }
    move_flying(p, strafe, forward, accel);
    move_entity(p, p.vx, p.vy, p.vz, ax, az);
    p.vy = (p.vy - GRAVITY) * AIR_DRAG_V;
    p.vx *= f4;
    p.vz *= f4;
}

// --------------------------------------------------------------------- combat
__device__ inline double ray_aabb(double ox, double oy, double oz,
                                  double dxr, double dyr, double dzr,
                                  double mnx, double mny, double mnz,
                                  double mxx, double mxy, double mxz,
                                  double max_dist) {
    if (ox >= mnx && ox <= mxx && oy >= mny && oy <= mxy && oz >= mnz && oz <= mxz)
        return 0.0;
    double t_min = 0.0, t_max = max_dist;
    double o[3] = {ox, oy, oz}, d[3] = {dxr, dyr, dzr};
    double lo[3] = {mnx, mny, mnz}, hi[3] = {mxx, mxy, mxz};
    for (int k = 0; k < 3; ++k) {
        if (fabs(d[k]) < 1.0e-12) {
            if (o[k] < lo[k] || o[k] > hi[k]) return -1.0;
        } else {
            double inv = 1.0 / d[k];
            double t1 = (lo[k] - o[k]) * inv, t2 = (hi[k] - o[k]) * inv;
            if (t1 > t2) { double t = t1; t1 = t2; t2 = t; }
            if (t1 > t_min) t_min = t1;
            if (t2 < t_max) t_max = t2;
            if (t_min > t_max) return -1.0;
        }
    }
    return t_min;
}

__device__ inline bool can_hit(const P &a, const P &t) {
    double ex = a.x, ey = a.y + PLAYER_EYE_HEIGHT, ez = a.z;
    double yr = a.yaw * DEG2RAD, pr = a.pitch * DEG2RAD;
    double cp = cos(pr);
    double lx = -sin(yr) * cp, ly = -sin(pr), lz = cos(yr) * cp;
    double b = COLLISION_BORDER, hw = PLAYER_HALF_WIDTH;
    double dist = ray_aabb(ex, ey, ez, lx, ly, lz,
                           t.x - hw - b, t.y - b, t.z - hw - b,
                           t.x + hw + b, t.y + PLAYER_HEIGHT + b, t.z + hw + b,
                           ATTACK_REACH);
    return dist >= 0.0;
}

__device__ inline void knock_back(P &t, double rx, double rz) {
    double f = sqrt(rx * rx + rz * rz);
    if (f < 1.0e-4) return;
    t.vx /= 2.0; t.vz /= 2.0;
    t.vx -= rx / f * KNOCKBACK_STRENGTH;
    t.vz -= rz / f * KNOCKBACK_STRENGTH;
    if (t.og) {
        t.vy /= 2.0;
        t.vy += KNOCKBACK_STRENGTH;
        if (t.vy > KNOCKBACK_Y_CAP) t.vy = KNOCKBACK_Y_CAP;
    }
}

// retourne 1 si hit comptabilisé
__device__ inline int try_attack(P &a, P &t) {
    if (a.ccd > 0) return 0;
    int cd = (int)(20.0 / a.h_cps + 0.5);
    a.ccd = cd < 1 ? 1 : cd;
    if (!can_hit(a, t)) return 0;
    if (t.hurt > HURT_REHIT) return 0;
    t.hurt = MAX_HURT;
    a.hits += 1;
    knock_back(t, a.x - t.x, a.z - t.z);
    if (a.spr) {
        double yr = a.yaw * DEG2RAD;
        t.vx += -sin(yr) * SPRINT_KB_H;
        t.vy += SPRINT_KB_Y;
        t.vz += cos(yr) * SPRINT_KB_H;
        a.vx *= ATTACKER_SLOWDOWN;
        a.vz *= ATTACKER_SLOWDOWN;
        a.spr = 0;
    }
    return 1;
}

__device__ inline void entity_push(P &a, P &b) {
    if (fabs(b.x - a.x) >= 2.0 * PLAYER_HALF_WIDTH + 0.2) return;
    if (fabs(b.z - a.z) >= 2.0 * PLAYER_HALF_WIDTH + 0.2) return;
    if (b.y >= a.y + PLAYER_HEIGHT || b.y + PLAYER_HEIGHT <= a.y) return;
    double d0 = b.x - a.x, d1 = b.z - a.z;
    double d2 = fmax(fabs(d0), fabs(d1));
    if (d2 >= 0.01) {
        d2 = sqrt(d2);
        d0 /= d2; d1 /= d2;
        double d3 = 1.0 / d2;
        if (d3 > 1.0) d3 = 1.0;
        d0 *= d3 * 0.05;
        d1 *= d3 * 0.05;
        a.vx -= d0; a.vz -= d1;
        b.vx += d0; b.vz += d1;
    }
}

// ---------------------------------------------------------------- observation
// Miroir exact de sim/obs.py::build_obs
__device__ inline void write_obs(float *o, const P &own, const P &opp,
                                 const SimParams &pr, const float *last_act,
                                 int tick) {
    double sy = sin(own.yaw * DEG2RAD), cy = cos(own.yaw * DEG2RAD);
    double dx = opp.x - own.x, dy = opp.y - own.y, dz = opp.z - own.z;
    double along = dx * -sy + dz * cy, side = dx * cy + dz * sy;
    double dist_h = sqrt(dx * dx + dz * dz);
    double dist3 = sqrt(dx * dx + dy * dy + dz * dz);
    double eye_dy = (opp.y + 0.9) - (own.y + PLAYER_EYE_HEIGHT);
    double yaw_to = dist_h > 1e-9 ? atan2(-dx, dz) * RAD2DEG : own.yaw;
    double yaw_err = wrap_degrees(yaw_to - own.yaw) * DEG2RAD;
    double pitch_to = dist_h > 1e-9 ? -atan2(eye_dy, dist_h) * RAD2DEG : 0.0;
    double pitch_err = pitch_to - own.pitch;
    double ova = opp.vx * -sy + opp.vz * cy, ovs = opp.vx * cy + opp.vz * sy;
    double sva = own.vx * -sy + own.vz * cy, svs = own.vx * cy + own.vz * sy;
    double dyaw_rel = wrap_degrees(opp.yaw - own.yaw) * DEG2RAD;

    o[0] = (float)(along / 8.0);
    o[1] = (float)(side / 8.0);
    o[2] = (float)(dy / 4.0);
    o[3] = (float)(dist3 / 8.0);
    o[4] = (float)ova;
    o[5] = (float)ovs;
    o[6] = (float)opp.vy;
    o[7] = (float)sva;
    o[8] = (float)svs;
    o[9] = (float)own.vy;
    o[10] = (float)(own.pitch / 90.0);
    o[11] = (float)sin(yaw_err);
    o[12] = (float)cos(yaw_err);
    o[13] = (float)(pitch_err / 90.0);
    o[14] = (float)sin(dyaw_rel);
    o[15] = (float)cos(dyaw_rel);
    o[16] = (float)(opp.pitch / 90.0);
    o[17] = own.og ? 1.0f : 0.0f;
    o[18] = opp.og ? 1.0f : 0.0f;
    o[19] = own.spr ? 1.0f : 0.0f;
    o[20] = opp.spr ? 1.0f : 0.0f;
    o[21] = (float)(own.hurt / 20.0);
    o[22] = (float)(opp.hurt / 20.0);
    o[23] = (float)(own.ccd / 20.0);
    o[24] = (float)(own.jt / 10.0);
    o[25] = (float)(((double)pr.arena_x - 0.3 - own.x) / 8.0);
    o[26] = (float)((own.x - 0.3) / 8.0);
    o[27] = (float)(((double)pr.arena_z - 0.3 - own.z) / 8.0);
    o[28] = (float)((own.z - 0.3) / 8.0);
    o[29] = (float)sy;
    o[30] = (float)cy;
    o[31] = (float)(own.hits / 100.0);
    o[32] = (float)(opp.hits / 100.0);
    o[33] = (float)((own.hits - opp.hits) / 20.0);
    o[34] = (float)(((double)pr.max_ticks - (double)tick) / (double)pr.max_ticks);
    o[35] = own.h_cps / 20.0f;
    o[36] = own.h_rot / 180.0f;
    o[37] = (float)own.h_delay / 8.0f;
    for (int k = 0; k < ACT_DIM; ++k) o[38 + k] = last_act[k];
    o[45] = (float)(dist_h / 8.0);
    o[46] = (float)(own.y / 4.0);
    o[47] = (float)(opp.y / 4.0);
}

// --------------------------------------------------------------- état mémoire
struct StatePtrs {
    double *pos;        // [N,2,8] x,y,z,vx,vy,vz,yaw,pitch
    int *ints;          // [N,2,8] hurt, jt, ccd, hits, og, spr, col, h_delay
    float *human;       // [N,2,2] h_cps, h_rot
    int *tick;          // [N]
    float *queue;       // [N,2,MAX_DELAY,ACT_DIM]
    float *last_act;    // [N,2,ACT_DIM]
    unsigned long long *rng;  // [N]
};

__device__ inline void load_agent(const StatePtrs &S, int n, int i, P &p) {
    const double *d = S.pos + ((long long)n * 2 + i) * 8;
    p.x = d[0]; p.y = d[1]; p.z = d[2];
    p.vx = d[3]; p.vy = d[4]; p.vz = d[5];
    p.yaw = d[6]; p.pitch = d[7];
    const int *q = S.ints + ((long long)n * 2 + i) * 8;
    p.hurt = q[0]; p.jt = q[1]; p.ccd = q[2]; p.hits = q[3];
    p.og = q[4]; p.spr = q[5]; p.col = q[6]; p.h_delay = q[7];
    const float *h = S.human + ((long long)n * 2 + i) * 2;
    p.h_cps = h[0]; p.h_rot = h[1];
}

__device__ inline void store_agent(const StatePtrs &S, int n, int i, const P &p) {
    double *d = S.pos + ((long long)n * 2 + i) * 8;
    d[0] = p.x; d[1] = p.y; d[2] = p.z;
    d[3] = p.vx; d[4] = p.vy; d[5] = p.vz;
    d[6] = p.yaw; d[7] = p.pitch;
    int *q = S.ints + ((long long)n * 2 + i) * 8;
    q[0] = p.hurt; q[1] = p.jt; q[2] = p.ccd; q[3] = p.hits;
    q[4] = p.og; q[5] = p.spr; q[6] = p.col; q[7] = p.h_delay;
    float *h = S.human + ((long long)n * 2 + i) * 2;
    h[0] = p.h_cps; h[1] = p.h_rot;
}

__device__ inline void reset_match(const StatePtrs &S, int n, const SimParams &pr,
                                   P *agents) {
    unsigned long long rs = S.rng[n];
    double cx = (double)pr.arena_x / 2.0, cz = (double)pr.arena_z / 2.0;
    double gap = fmin((double)pr.arena_x, (double)pr.arena_z) / 3.0;
    int randomize = pr.randomize > 0.5f;
    for (int i = 0; i < 2; ++i) {
        P p;
        p.x = cx; p.y = 0.0;
        p.z = i == 0 ? cz - gap : cz + gap;
        p.yaw = i == 0 ? 0.0 : 180.0;
        p.pitch = 0.0;
        p.vx = p.vy = p.vz = 0.0;
        p.hurt = p.jt = p.ccd = p.hits = 0;
        p.og = 1; p.spr = 0; p.col = 0;
        if (randomize) {
            p.x += (rng_next(rs) * 2.0 - 1.0) * (double)pr.jitter;
            p.z += (rng_next(rs) * 2.0 - 1.0) * (double)pr.jitter;
            p.x = clampd(p.x, 0.3, (double)pr.arena_x - 0.3);
            p.z = clampd(p.z, 0.3, (double)pr.arena_z - 0.3);
            p.h_cps = pr.cps_min + (float)rng_next(rs) * (pr.cps_max - pr.cps_min);
            p.h_rot = pr.rot_min + (float)rng_next(rs) * (pr.rot_max - pr.rot_min);
            p.h_delay = (int)(pr.delay_min
                        + rng_next(rs) * ((double)pr.delay_max - (double)pr.delay_min) + 0.5);
        } else {
            p.h_cps = (pr.cps_min + pr.cps_max) * 0.5f;
            p.h_rot = (pr.rot_min + pr.rot_max) * 0.5f;
            p.h_delay = (int)(((double)pr.delay_min + (double)pr.delay_max) * 0.5 + 0.5);
        }
        if (p.h_delay > MAX_DELAY - 1) p.h_delay = MAX_DELAY - 1;
        agents[i] = p;
    }
    S.rng[n] = rs;
    S.tick[n] = 0;
    float *q = S.queue + (long long)n * 2 * MAX_DELAY * ACT_DIM;
    for (int k = 0; k < 2 * MAX_DELAY * ACT_DIM; ++k) q[k] = 0.0f;
    float *la = S.last_act + (long long)n * 2 * ACT_DIM;
    for (int k = 0; k < 2 * ACT_DIM; ++k) la[k] = 0.0f;
}

// -------------------------------------------------------------------- kernels
__global__ void k_reset(StatePtrs S, SimParams pr, float *obs, int n_envs,
                        unsigned long long seed) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_envs) return;
    S.rng[n] = seed + (unsigned long long)n * 0x9E3779B97F4A7C15ULL + 1ULL;
    // chauffe du xorshift
    rng_next(S.rng[n]); rng_next(S.rng[n]);
    P agents[2];
    reset_match(S, n, pr, agents);
    store_agent(S, n, 0, agents[0]);
    store_agent(S, n, 1, agents[1]);
    for (int i = 0; i < 2; ++i)
        write_obs(obs + ((long long)n * 2 + i) * OBS_DIM,
                  agents[i], agents[1 - i], pr,
                  S.last_act + ((long long)n * 2 + i) * ACT_DIM, 0);
}

__global__ void k_tick(StatePtrs S, SimParams pr, const float *actions,
                       float *obs, float *reward, unsigned char *done,
                       int *winner, int n_envs) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_envs) return;

    P pl[2];
    load_agent(S, n, 0, pl[0]);
    load_agent(S, n, 1, pl[1]);
    int tick = S.tick[n];

    // file circulaire de latence + mémorisation de l'action décidée
    float applied[2][ACT_DIM];
    for (int i = 0; i < 2; ++i) {
        const float *in = actions + ((long long)n * 2 + i) * ACT_DIM;
        float *slot = S.queue + (((long long)n * 2 + i) * MAX_DELAY
                                 + (tick % MAX_DELAY)) * ACT_DIM;
        float *la = S.last_act + ((long long)n * 2 + i) * ACT_DIM;
        for (int k = 0; k < ACT_DIM; ++k) { slot[k] = in[k]; la[k] = in[k]; }
        int d = pl[i].h_delay;
        int rd = ((tick - d) % MAX_DELAY + MAX_DELAY) % MAX_DELAY;
        const float *src = (tick >= d)
            ? S.queue + (((long long)n * 2 + i) * MAX_DELAY + rd) * ACT_DIM
            : nullptr;
        for (int k = 0; k < ACT_DIM; ++k) applied[i][k] = src ? src[k] : 0.0f;
    }

    // décode les actions
    double dyaw[2], dpitch[2];
    int fwd[2], strafe[2], jmp[2], spr_key[2], atk[2];
    for (int i = 0; i < 2; ++i) {
        dyaw[i] = clampd((double)applied[i][0], -1.0, 1.0) * (double)pl[i].h_rot;
        dpitch[i] = clampd((double)applied[i][1], -1.0, 1.0) * (double)pl[i].h_rot;
        fwd[i] = applied[i][2] > 0.5f ? 1 : (applied[i][2] < -0.5f ? -1 : 0);
        strafe[i] = applied[i][3] > 0.5f ? 1 : (applied[i][3] < -0.5f ? -1 : 0);
        jmp[i] = applied[i][4] > 0.5f;
        spr_key[i] = applied[i][5] > 0.5f;
        atk[i] = applied[i][6] > 0.5f;
    }

    // 1. timers
    for (int i = 0; i < 2; ++i) {
        if (pl[i].hurt > 0) pl[i].hurt -= 1;
        if (pl[i].ccd > 0) pl[i].ccd -= 1;
    }
    // 2. rotations (le clamp ±h_rot est déjà fait via la normalisation [-1,1])
    for (int i = 0; i < 2; ++i) {
        pl[i].yaw += dyaw[i];
        pl[i].pitch = clampd(pl[i].pitch + dpitch[i], -90.0, 90.0);
    }
    // 3. sprint
    for (int i = 0; i < 2; ++i)
        pl[i].spr = (spr_key[i] && fwd[i] > 0 && !pl[i].col) ? 1 : 0;

    // 4. attaques (séquentiel agent 0 puis 1, comme sim_ref)
    int dealt[2] = {0, 0};
    for (int i = 0; i < 2; ++i)
        if (atk[i]) dealt[i] = try_attack(pl[i], pl[1 - i]);

    // 5. mouvement
    for (int i = 0; i < 2; ++i)
        living_update_movement(pl[i], (double)strafe[i], (double)fwd[i], jmp[i],
                               (double)pr.arena_x, (double)pr.arena_z, (int)pr.amp);

    // 5b. poussée entre joueurs (2x, comme vanilla)
    entity_push(pl[0], pl[1]);
    entity_push(pl[1], pl[0]);

    // 6. règles boxing + reward
    tick += 1;
    S.tick[n] = tick;
    float rw[2];
    for (int i = 0; i < 2; ++i) {
        rw[i] = pr.r_hit * (float)dealt[i] + pr.r_hurt * (float)dealt[1 - i];
        if (pr.r_dist != 0.0f) {
            double ddx = pl[i].x - pl[1 - i].x, ddy = pl[i].y - pl[1 - i].y,
                   ddz = pl[i].z - pl[1 - i].z;
            rw[i] -= pr.r_dist * (float)sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
        }
    }
    int win = -2;
    if (pl[0].hits >= (int)pr.target_hits) win = 0;
    if (pl[1].hits >= (int)pr.target_hits) win = 1;
    if (win == -2 && tick >= (int)pr.max_ticks) {
        win = pl[0].hits > pl[1].hits ? 0 : (pl[1].hits > pl[0].hits ? 1 : -1);
    }

    bool is_done = win != -2;
    if (is_done && win >= 0) {
        rw[win] += pr.r_win;
        rw[1 - win] -= pr.r_win;
    }
    reward[(long long)n * 2 + 0] = rw[0];
    reward[(long long)n * 2 + 1] = rw[1];
    done[n] = is_done ? 1 : 0;
    winner[n] = is_done ? win : -2;

    if (is_done) {
        reset_match(S, n, pr, pl);   // auto-reset (obs = 1er tick du nouveau match)
        tick = 0;
    }
    store_agent(S, n, 0, pl[0]);
    store_agent(S, n, 1, pl[1]);
    for (int i = 0; i < 2; ++i)
        write_obs(obs + ((long long)n * 2 + i) * OBS_DIM,
                  pl[i], pl[1 - i], pr,
                  S.last_act + ((long long)n * 2 + i) * ACT_DIM, tick);
}

// ------------------------------------------------------------------- bindings
static SimParams params_from_vec(const std::vector<double> &v) {
    TORCH_CHECK(v.size() == 17, "SimParams: 17 valeurs attendues");
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
    return p;
}

static StatePtrs ptrs_from_tensors(torch::Tensor pos, torch::Tensor ints,
                                   torch::Tensor human, torch::Tensor tick,
                                   torch::Tensor queue, torch::Tensor last_act,
                                   torch::Tensor rng) {
    StatePtrs S;
    S.pos = pos.data_ptr<double>();
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
