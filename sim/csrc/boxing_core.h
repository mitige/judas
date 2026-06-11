// Judas — cœur du simulateur boxing 1.8.9.
//
// Compile en CUDA (boxing_kernel.cu) et en C++ pur (tools/cpu_check.cpp).
//
// PRÉCISION : jreal = float par défaut (vitesse maximale sur GPU grand
// public : le FP64 y est ~32x plus lent). Compiler avec -DJUDAS_DOUBLE pour
// la précision double exacte (vérification d'équivalence contre sim_ref).
//
// SOURCE DE VÉRITÉ de la physique : sim_ref/ (Python).

#pragma once

#include <math.h>
#ifndef __CUDACC__
#include <cmath>
#endif

#ifdef __CUDACC__
#define JD __device__ __forceinline__
#else
#define JD inline
#endif

#ifdef JUDAS_DOUBLE
typedef double jreal;
#else
typedef float jreal;
#endif

// wrappers maths : résolus vers l'overload float ou double selon jreal
JD jreal jsin(jreal x)  {
#ifdef __CUDACC__
    return sin(x);
#else
    return std::sin(x);
#endif
}
JD jreal jcos(jreal x)  {
#ifdef __CUDACC__
    return cos(x);
#else
    return std::cos(x);
#endif
}
JD jreal jsqrt(jreal x) {
#ifdef __CUDACC__
    return sqrt(x);
#else
    return std::sqrt(x);
#endif
}
JD jreal jabs(jreal x)  {
#ifdef __CUDACC__
    return fabs(x);
#else
    return std::fabs(x);
#endif
}
JD jreal jmod(jreal x, jreal y) {
#ifdef __CUDACC__
    return fmod(x, y);
#else
    return std::fmod(x, y);
#endif
}
JD jreal jatan2(jreal y, jreal x) {
#ifdef __CUDACC__
    return atan2(y, x);
#else
    return std::atan2(y, x);
#endif
}

#define MAX_DELAY 8
#define ACT_DIM 7
#define OBS_DIM 48

// ------------------------------------------------------------- constantes 1.8.9
#define PLAYER_HALF_WIDTH ((jreal)0.3)
#define PLAYER_HEIGHT ((jreal)1.8)
#define PLAYER_EYE_HEIGHT ((jreal)1.62)
#define INPUT_FACTOR ((jreal)0.98)
#define GROUND_FRICTION ((jreal)0.546)
#define AIR_DRAG_H ((jreal)0.91)
#define MAGIC_GROUND ((jreal)0.16277136)
#define GRAVITY ((jreal)0.08)
#define AIR_DRAG_V ((jreal)0.98)
#define JUMP_MOTION_Y ((jreal)0.42)
#define SPRINT_JUMP_BOOST ((jreal)0.2)
#define JUMP_COOLDOWN_TICKS 10
#define AIR_MOVE_FACTOR ((jreal)0.02)
#define BASE_MOVE_SPEED ((jreal)0.10000000149011612)
#define SPRINT_MODIFIER ((jreal)0.3)
#define SPEED_POTION_PER_LEVEL ((jreal)0.20000000298023224)
#define ATTACK_REACH ((jreal)3.0)
#define COLLISION_BORDER ((jreal)0.1)
#define KNOCKBACK_STRENGTH ((jreal)0.4)
#define KNOCKBACK_Y_CAP ((jreal)0.4)
#define SPRINT_KB_H ((jreal)0.5)
#define SPRINT_KB_Y ((jreal)0.1)
#define ATTACKER_SLOWDOWN ((jreal)0.6)
#define MAX_HURT 20
#define HURT_REHIT 10
#define DEG2RAD ((jreal)0.017453292519943295)
#define RAD2DEG ((jreal)57.29577951308232)

#define R0 ((jreal)0.0)
#define R1 ((jreal)1.0)
#define R2 ((jreal)2.0)
#define RHALF ((jreal)0.5)

struct SimParams {
    float arena_x, arena_z;
    float target_hits, max_ticks, amp;
    float cps_min, cps_max, rot_min, rot_max, delay_min, delay_max;
    float jitter;
    float r_hit, r_hurt, r_win, r_dist;
    float randomize;
    float spawn_gap;     // demi-distance de spawn (0 = arène/3)
    float kb_h, kb_v, kb_idle;   // knockback custom (1.0 = vanilla)
    float r_combo, combo_window, combo_cap;   // bonus combo (0 = off)
};

// État d'un agent en registres
struct P {
    jreal x, y, z, vx, vy, vz, yaw, pitch;
    int hurt, jt, ccd, hits;
    int og, spr, col;          // on_ground, sprinting, collided_horizontally
    int combo, last_hit;       // chaîne de hits portés, tick du dernier hit
    float h_cps, h_rot;        // humanisation
    int h_delay;
};

// ----------------------------------------------------------------------- RNG
JD double rng_next(unsigned long long &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (double)(s >> 11) * (1.0 / 9007199254740992.0);  // [0,1)
}

JD jreal clampr(jreal v, jreal lo, jreal hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

JD jreal wrap_degrees(jreal a) {
    a = jmod(a, (jreal)360.0);
    if (a >= (jreal)180.0) a -= (jreal)360.0;
    if (a < (jreal)-180.0) a += (jreal)360.0;
    return a;
}

// ------------------------------------------------------------------ mouvement
JD jreal move_speed(int sprinting, int amp) {
    jreal s = BASE_MOVE_SPEED;
    if (amp >= 0) s *= R1 + SPEED_POTION_PER_LEVEL * (jreal)(amp + 1);
    if (sprinting) s *= R1 + SPRINT_MODIFIER;
    return s;
}

JD void do_jump(P &p) {
    p.vy = JUMP_MOTION_Y;
    if (p.spr) {
        jreal yr = p.yaw * DEG2RAD;
        p.vx -= jsin(yr) * SPRINT_JUMP_BOOST;
        p.vz += jcos(yr) * SPRINT_JUMP_BOOST;
    }
}

JD void move_flying(P &p, jreal strafe, jreal forward, jreal friction) {
    jreal f = strafe * strafe + forward * forward;
    if (f >= (jreal)1.0e-4) {
        f = jsqrt(f);
        if (f < R1) f = R1;
        f = friction / f;
        strafe *= f; forward *= f;
        jreal s = jsin(p.yaw * DEG2RAD), c = jcos(p.yaw * DEG2RAD);
        p.vx += strafe * c - forward * s;
        p.vz += forward * c + strafe * s;
    }
}

JD void move_entity(P &p, jreal dx, jreal dy, jreal dz, jreal ax, jreal az) {
    jreal dx0 = dx, dy0 = dy, dz0 = dz;
    jreal ny = p.y + dy;
    if (ny < R0) { ny = R0; dy = ny - p.y; }
    p.y = ny;
    jreal lo = PLAYER_HALF_WIDTH, hx = ax - PLAYER_HALF_WIDTH;
    jreal nx = p.x + dx;
    if (nx < lo) { nx = lo; dx = nx - p.x; }
    else if (nx > hx) { nx = hx; dx = nx - p.x; }
    p.x = nx;
    jreal hz = az - PLAYER_HALF_WIDTH;
    jreal nz = p.z + dz;
    if (nz < lo) { nz = lo; dz = nz - p.z; }
    else if (nz > hz) { nz = hz; dz = nz - p.z; }
    p.z = nz;
    bool cx = dx0 != dx, cy = dy0 != dy, cz = dz0 != dz;
    p.col = (cx || cz) ? 1 : 0;
    p.og = (cy && dy0 < R0) ? 1 : 0;
    if (cx) p.vx = R0;
    if (cy) p.vy = R0;
    if (cz) p.vz = R0;
}

JD void living_update_movement(P &p, jreal strafe_in, jreal forward_in,
                               int jumping, jreal ax, jreal az, int amp) {
    jreal strafe = strafe_in * INPUT_FACTOR;
    jreal forward = forward_in * INPUT_FACTOR;
    if (p.jt > 0) p.jt -= 1;
    if (jumping) {
        if (p.og && p.jt == 0) { do_jump(p); p.jt = JUMP_COOLDOWN_TICKS; }
    } else p.jt = 0;

    jreal f4, accel;
    if (p.og) {
        f4 = GROUND_FRICTION;
        accel = move_speed(p.spr, amp) * (MAGIC_GROUND / (f4 * f4 * f4));
    } else {
        f4 = AIR_DRAG_H;
        accel = AIR_MOVE_FACTOR * (p.spr ? (jreal)1.3 : R1);
    }
    move_flying(p, strafe, forward, accel);
    move_entity(p, p.vx, p.vy, p.vz, ax, az);
    p.vy = (p.vy - GRAVITY) * AIR_DRAG_V;
    p.vx *= f4;
    p.vz *= f4;
}

// --------------------------------------------------------------------- combat
JD jreal ray_aabb(jreal ox, jreal oy, jreal oz,
                  jreal dxr, jreal dyr, jreal dzr,
                  jreal mnx, jreal mny, jreal mnz,
                  jreal mxx, jreal mxy, jreal mxz,
                  jreal max_dist) {
    if (ox >= mnx && ox <= mxx && oy >= mny && oy <= mxy && oz >= mnz && oz <= mxz)
        return R0;
    jreal t_min = R0, t_max = max_dist;
    jreal o[3] = {ox, oy, oz}, d[3] = {dxr, dyr, dzr};
    jreal lo[3] = {mnx, mny, mnz}, hi[3] = {mxx, mxy, mxz};
    for (int k = 0; k < 3; ++k) {
        if (jabs(d[k]) < (jreal)1.0e-12) {
            if (o[k] < lo[k] || o[k] > hi[k]) return (jreal)-1.0;
        } else {
            jreal inv = R1 / d[k];
            jreal t1 = (lo[k] - o[k]) * inv, t2 = (hi[k] - o[k]) * inv;
            if (t1 > t2) { jreal t = t1; t1 = t2; t2 = t; }
            if (t1 > t_min) t_min = t1;
            if (t2 < t_max) t_max = t2;
            if (t_min > t_max) return (jreal)-1.0;
        }
    }
    return t_min;
}

JD bool can_hit(const P &a, const P &t) {
    jreal ex = a.x, ey = a.y + PLAYER_EYE_HEIGHT, ez = a.z;
    jreal yr = a.yaw * DEG2RAD, pr = a.pitch * DEG2RAD;
    jreal cp = jcos(pr);
    jreal lx = -jsin(yr) * cp, ly = -jsin(pr), lz = jcos(yr) * cp;
    jreal b = COLLISION_BORDER, hw = PLAYER_HALF_WIDTH;
    jreal dist = ray_aabb(ex, ey, ez, lx, ly, lz,
                          t.x - hw - b, t.y - b, t.z - hw - b,
                          t.x + hw + b, t.y + PLAYER_HEIGHT + b, t.z + hw + b,
                          ATTACK_REACH);
    return dist >= R0;
}

JD void knock_back(P &t, jreal rx, jreal rz, jreal kb_h, jreal kb_v) {
    jreal f = jsqrt(rx * rx + rz * rz);
    if (f < (jreal)1.0e-4) return;
    t.vx /= R2; t.vz /= R2;
    t.vx -= rx / f * KNOCKBACK_STRENGTH * kb_h;
    t.vz -= rz / f * KNOCKBACK_STRENGTH * kb_h;
    if (t.og) {
        t.vy /= R2;
        t.vy += KNOCKBACK_STRENGTH * kb_v;
        if (t.vy > KNOCKBACK_Y_CAP * kb_v) t.vy = KNOCKBACK_Y_CAP * kb_v;
    }
}

// retourne 1 si hit comptabilisé. kb custom : 1.0 = vanilla exact
JD int try_attack(P &a, P &t, jreal kb_h, jreal kb_v, jreal kb_idle,
                  int target_idle) {
    if (a.ccd > 0) return 0;
    int cd = (int)(20.0 / (double)a.h_cps + 0.5);
    a.ccd = cd < 1 ? 1 : cd;
    if (!can_hit(a, t)) return 0;
    if (t.hurt > HURT_REHIT) return 0;
    t.hurt = MAX_HURT;
    a.hits += 1;
    jreal eff_h = kb_h * (target_idle ? kb_idle : R1);
    jreal eff_v = kb_v * (target_idle ? kb_idle : R1);
    knock_back(t, a.x - t.x, a.z - t.z, eff_h, eff_v);
    if (a.spr) {
        jreal yr = a.yaw * DEG2RAD;
        t.vx += -jsin(yr) * SPRINT_KB_H * eff_h;
        t.vy += SPRINT_KB_Y * eff_v;
        t.vz += jcos(yr) * SPRINT_KB_H * eff_h;
        a.vx *= ATTACKER_SLOWDOWN;
        a.vz *= ATTACKER_SLOWDOWN;
        a.spr = 0;
    }
    return 1;
}

JD void entity_push(P &a, P &b) {
    if (jabs(b.x - a.x) >= R2 * PLAYER_HALF_WIDTH + (jreal)0.2) return;
    if (jabs(b.z - a.z) >= R2 * PLAYER_HALF_WIDTH + (jreal)0.2) return;
    if (b.y >= a.y + PLAYER_HEIGHT || b.y + PLAYER_HEIGHT <= a.y) return;
    jreal d0 = b.x - a.x, d1 = b.z - a.z;
    jreal d2 = jabs(d0) > jabs(d1) ? jabs(d0) : jabs(d1);
    if (d2 >= (jreal)0.01) {
        d2 = jsqrt(d2);
        d0 /= d2; d1 /= d2;
        jreal d3 = R1 / d2;
        if (d3 > R1) d3 = R1;
        d0 *= d3 * (jreal)0.05;
        d1 *= d3 * (jreal)0.05;
        a.vx -= d0; a.vz -= d1;
        b.vx += d0; b.vz += d1;
    }
}

// ---------------------------------------------------------------- observation
// Miroir exact de sim/obs.py::build_obs
JD void write_obs(float *o, const P &own, const P &opp,
                  const SimParams &pr, const float *last_act, int tick) {
    jreal sy = jsin(own.yaw * DEG2RAD), cy = jcos(own.yaw * DEG2RAD);
    jreal dx = opp.x - own.x, dy = opp.y - own.y, dz = opp.z - own.z;
    jreal along = dx * -sy + dz * cy, side = dx * cy + dz * sy;
    jreal dist_h = jsqrt(dx * dx + dz * dz);
    jreal dist3 = jsqrt(dx * dx + dy * dy + dz * dz);
    jreal eye_dy = (opp.y + (jreal)0.9) - (own.y + PLAYER_EYE_HEIGHT);
    jreal yaw_to = dist_h > (jreal)1e-9 ? jatan2(-dx, dz) * RAD2DEG : own.yaw;
    jreal yaw_err = wrap_degrees(yaw_to - own.yaw) * DEG2RAD;
    jreal pitch_to = dist_h > (jreal)1e-9 ? -jatan2(eye_dy, dist_h) * RAD2DEG : R0;
    jreal pitch_err = pitch_to - own.pitch;
    jreal ova = opp.vx * -sy + opp.vz * cy, ovs = opp.vx * cy + opp.vz * sy;
    jreal sva = own.vx * -sy + own.vz * cy, svs = own.vx * cy + own.vz * sy;
    jreal dyaw_rel = wrap_degrees(opp.yaw - own.yaw) * DEG2RAD;

    o[0] = (float)(along / (jreal)8.0);
    o[1] = (float)(side / (jreal)8.0);
    o[2] = (float)(dy / (jreal)4.0);
    o[3] = (float)(dist3 / (jreal)8.0);
    o[4] = (float)ova;
    o[5] = (float)ovs;
    o[6] = (float)opp.vy;
    o[7] = (float)sva;
    o[8] = (float)svs;
    o[9] = (float)own.vy;
    o[10] = (float)(own.pitch / (jreal)90.0);
    o[11] = (float)jsin(yaw_err);
    o[12] = (float)jcos(yaw_err);
    o[13] = (float)(pitch_err / (jreal)90.0);
    o[14] = (float)jsin(dyaw_rel);
    o[15] = (float)jcos(dyaw_rel);
    o[16] = (float)(opp.pitch / (jreal)90.0);
    o[17] = own.og ? 1.0f : 0.0f;
    o[18] = opp.og ? 1.0f : 0.0f;
    o[19] = own.spr ? 1.0f : 0.0f;
    o[20] = opp.spr ? 1.0f : 0.0f;
    o[21] = (float)own.hurt / 20.0f;
    o[22] = (float)opp.hurt / 20.0f;
    o[23] = (float)own.ccd / 20.0f;
    o[24] = (float)own.jt / 10.0f;
    o[25] = (float)(((jreal)pr.arena_x - (jreal)0.3 - own.x) / (jreal)8.0);
    o[26] = (float)((own.x - (jreal)0.3) / (jreal)8.0);
    o[27] = (float)(((jreal)pr.arena_z - (jreal)0.3 - own.z) / (jreal)8.0);
    o[28] = (float)((own.z - (jreal)0.3) / (jreal)8.0);
    o[29] = (float)sy;
    o[30] = (float)cy;
    o[31] = (float)own.hits / 100.0f;
    o[32] = (float)opp.hits / 100.0f;
    o[33] = (float)(own.hits - opp.hits) / 20.0f;
    o[34] = (float)(((double)pr.max_ticks - (double)tick) / (double)pr.max_ticks);
    o[35] = own.h_cps / 20.0f;
    o[36] = own.h_rot / 180.0f;
    o[37] = (float)own.h_delay / 8.0f;
    for (int k = 0; k < ACT_DIM; ++k) o[38 + k] = last_act[k];
    o[45] = (float)(dist_h / (jreal)8.0);
    o[46] = (float)(own.y / (jreal)4.0);
    o[47] = (float)(opp.y / (jreal)4.0);
}

// --------------------------------------------------------------- état mémoire
struct StatePtrs {
    jreal *pos;         // [N,2,8] x,y,z,vx,vy,vz,yaw,pitch
    int *ints;          // [N,2,10] hurt, jt, ccd, hits, og, spr, col, h_delay, combo, last_hit
    float *human;       // [N,2,2] h_cps, h_rot
    int *tick;          // [N]
    float *queue;       // [N,2,MAX_DELAY,ACT_DIM]
    float *last_act;    // [N,2,ACT_DIM]
    unsigned long long *rng;  // [N]
};

JD void load_agent(const StatePtrs &S, int n, int i, P &p) {
    const jreal *d = S.pos + ((long long)n * 2 + i) * 8;
    p.x = d[0]; p.y = d[1]; p.z = d[2];
    p.vx = d[3]; p.vy = d[4]; p.vz = d[5];
    p.yaw = d[6]; p.pitch = d[7];
    const int *q = S.ints + ((long long)n * 2 + i) * 10;
    p.hurt = q[0]; p.jt = q[1]; p.ccd = q[2]; p.hits = q[3];
    p.og = q[4]; p.spr = q[5]; p.col = q[6]; p.h_delay = q[7];
    p.combo = q[8]; p.last_hit = q[9];
    const float *h = S.human + ((long long)n * 2 + i) * 2;
    p.h_cps = h[0]; p.h_rot = h[1];
}

JD void store_agent(const StatePtrs &S, int n, int i, const P &p) {
    jreal *d = S.pos + ((long long)n * 2 + i) * 8;
    d[0] = p.x; d[1] = p.y; d[2] = p.z;
    d[3] = p.vx; d[4] = p.vy; d[5] = p.vz;
    d[6] = p.yaw; d[7] = p.pitch;
    int *q = S.ints + ((long long)n * 2 + i) * 10;
    q[0] = p.hurt; q[1] = p.jt; q[2] = p.ccd; q[3] = p.hits;
    q[4] = p.og; q[5] = p.spr; q[6] = p.col; q[7] = p.h_delay;
    q[8] = p.combo; q[9] = p.last_hit;
    float *h = S.human + ((long long)n * 2 + i) * 2;
    h[0] = p.h_cps; h[1] = p.h_rot;
}

JD void reset_match(const StatePtrs &S, int n, const SimParams &pr, P *agents) {
    unsigned long long rs = S.rng[n];
    jreal cx = (jreal)pr.arena_x / R2, cz = (jreal)pr.arena_z / R2;
    jreal gap = pr.spawn_gap > 0.0f ? (jreal)pr.spawn_gap
        : ((jreal)pr.arena_x < (jreal)pr.arena_z ? (jreal)pr.arena_x
                                                 : (jreal)pr.arena_z) / (jreal)3.0;
    int randomize = pr.randomize > 0.5f;
    for (int i = 0; i < 2; ++i) {
        P p;
        p.x = cx; p.y = R0;
        p.z = i == 0 ? cz - gap : cz + gap;
        p.yaw = i == 0 ? R0 : (jreal)180.0;
        p.pitch = R0;
        p.vx = p.vy = p.vz = R0;
        p.hurt = p.jt = p.ccd = p.hits = 0;
        p.og = 1; p.spr = 0; p.col = 0;
        p.combo = 0; p.last_hit = 0;
        if (randomize) {
            p.x += (jreal)((rng_next(rs) * 2.0 - 1.0) * (double)pr.jitter);
            p.z += (jreal)((rng_next(rs) * 2.0 - 1.0) * (double)pr.jitter);
            p.x = clampr(p.x, (jreal)0.3, (jreal)pr.arena_x - (jreal)0.3);
            p.z = clampr(p.z, (jreal)0.3, (jreal)pr.arena_z - (jreal)0.3);
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

// ------------------------------------------------- corps des kernels (1 match)
JD void reset_one(const StatePtrs &S, const SimParams &pr, float *obs,
                  int n, unsigned long long seed) {
    S.rng[n] = seed + (unsigned long long)n * 0x9E3779B97F4A7C15ULL + 1ULL;
    rng_next(S.rng[n]); rng_next(S.rng[n]);   // chauffe du xorshift
    P agents[2];
    reset_match(S, n, pr, agents);
    store_agent(S, n, 0, agents[0]);
    store_agent(S, n, 1, agents[1]);
    for (int i = 0; i < 2; ++i)
        write_obs(obs + ((long long)n * 2 + i) * OBS_DIM,
                  agents[i], agents[1 - i], pr,
                  S.last_act + ((long long)n * 2 + i) * ACT_DIM, 0);
}

JD void tick_one(const StatePtrs &S, const SimParams &pr, const float *actions,
                 float *obs, float *reward, unsigned char *done, int *winner,
                 int n) {
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
            : (const float *)0;
        for (int k = 0; k < ACT_DIM; ++k) applied[i][k] = src ? src[k] : 0.0f;
    }

    // décode les actions
    jreal dyaw[2], dpitch[2];
    int fwd[2], strafe[2], jmp[2], spr_key[2], atk[2];
    for (int i = 0; i < 2; ++i) {
        dyaw[i] = clampr((jreal)applied[i][0], (jreal)-1.0, R1) * (jreal)pl[i].h_rot;
        dpitch[i] = clampr((jreal)applied[i][1], (jreal)-1.0, R1) * (jreal)pl[i].h_rot;
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
    // 2. rotations
    for (int i = 0; i < 2; ++i) {
        pl[i].yaw += dyaw[i];
        pl[i].pitch = clampr(pl[i].pitch + dpitch[i], (jreal)-90.0, (jreal)90.0);
    }
    // 3. sprint
    for (int i = 0; i < 2; ++i)
        pl[i].spr = (spr_key[i] && fwd[i] > 0 && !pl[i].col) ? 1 : 0;

    // 4. attaques (séquentiel agent 0 puis 1, comme sim_ref)
    int dealt[2] = {0, 0};
    for (int i = 0; i < 2; ++i)
        if (atk[i])
            dealt[i] = try_attack(pl[i], pl[1 - i],
                                  (jreal)pr.kb_h, (jreal)pr.kb_v,
                                  (jreal)pr.kb_idle,
                                  fwd[1 - i] == 0 && strafe[1 - i] == 0);

    // 5. mouvement
    for (int i = 0; i < 2; ++i)
        living_update_movement(pl[i], (jreal)strafe[i], (jreal)fwd[i], jmp[i],
                               (jreal)pr.arena_x, (jreal)pr.arena_z, (int)pr.amp);

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
            jreal ddx = pl[i].x - pl[1 - i].x, ddy = pl[i].y - pl[1 - i].y,
                  ddz = pl[i].z - pl[1 - i].z;
            rw[i] -= pr.r_dist * (float)jsqrt(ddx * ddx + ddy * ddy + ddz * ddz);
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
        reset_match(S, n, pr, pl);   // auto-reset
        tick = 0;
    }
    store_agent(S, n, 0, pl[0]);
    store_agent(S, n, 1, pl[1]);
    for (int i = 0; i < 2; ++i)
        write_obs(obs + ((long long)n * 2 + i) * OBS_DIM,
                  pl[i], pl[1 - i], pr,
                  S.last_act + ((long long)n * 2 + i) * ACT_DIM, tick);
}
