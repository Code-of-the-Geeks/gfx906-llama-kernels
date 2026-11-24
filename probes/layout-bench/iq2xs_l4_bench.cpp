// iq2xs_l4_bench.cpp — spike step 9: "spill RAM to reduce compute" for IQ2_XS batched decode and prefill on gfx906 (MI50).
//
// L4 = pre-dequantized weights: a signed int8 plane (the sign-applied iq2xs_grid values, 256 B per 256-weight block) plus a scale
// plane with one dword per 32-weight sub-block = fp16 d | scale byte << 16 (32 B per block). 288 B/block = 3.9x the 74 B stock block.
// The scale plane keeps d and the 4-bit scales instead of a fused fp16 scale, so the kernels can apply stock's integer formula
//   sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4,   acc += (d * d8) * sumi
// and be bit-exact against the stock-format path (a fused fp16 scale cannot be; see step 5 L3).
//
// Layout/variants (same compute core per kernel family, only the weight path differs):
//   S   stock block_iq2_xs (74 B), stock sign math (unpack_ksigns + vcmpne4 + xor + vsub4)
//   Sf  stock block_iq2_xs, fast sign (step 6/7 computed byte mask), bit-exact
//   L4  int8 plane + (d, sc) plane: no grid gather, no sign math
// Kernel families:
//   gemv<NC>  n = 1..8 columns, shaped like llama.cpp mmvq on GCN (nwarps 2 for n<=4 else 1, 1 row/block at n=1 else 2, 8 threads per
//             256-weight block, 32 weights per thread step); the weight decode is done once per step and shared by all columns.
//   gemm      n = 512 (prefill), 64x64 output tile per 256-thread block, K step 128 (4 sub-blocks), weights decoded (S/Sf) or copied
//             (L4) into an int8 LDS tile, q8_1 activations into LDS, each thread 4 rows x 4 cols, same per-sub-block integer formula.
// Correctness: every kernel vs a double CPU oracle (dequantize_row_iq2_xs semantics x the same q8_1 activations); Sf / L4 bitwise vs S.
//
// Build: build_l4.sh.   Run: run_l4.sh (same tensor lookup as run.sh).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

#include "iq2xs_grid.h"

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

#define QK_K 256
struct block_iq2_xs { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; };
static_assert(sizeof(block_iq2_xs) == 74, "block_iq2_xs size");
struct block_q8_1 { __half2 ds; int8_t qs[32]; };
static_assert(sizeof(block_q8_1) == 36, "q8_1 size");

__device__ uint64_t d_grid[512];

// ---- llama.cpp HIP helpers (vendors/hip.h, vecdotq.cuh, common.cuh), verbatim semantics ----
static __device__ __forceinline__ int vsub4(const int a, const int b) {
    const unsigned int a_large = a | 0x80808080;
    const unsigned int b_small = b & 0x7f7f7f7f;
    const unsigned int result_low_7bits = a_large - b_small;
    const unsigned int should_flip_high_1bit = (a ^ ~b) & 0x80808080;
    return result_low_7bits ^ should_flip_high_1bit;
}
static __device__ __forceinline__ unsigned int vcmpne4(unsigned int a, unsigned int b) {
    const unsigned int x = a ^ b;
    const unsigned int ne_low_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    const unsigned int ne_high_1bit = x & 0x80808080;
    return ((ne_low_7bits | ne_high_1bit) >> 7) * 0xff;
}
static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
static __device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }
static __device__ __forceinline__ int get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) (x16[2*i32 + 0] | ((uint32_t) x16[2*i32 + 1] << 16));
}
static __device__ __forceinline__ uint32_t mul_u24(uint32_t a, uint32_t b) {
    uint32_t r; asm("v_mul_u32_u24 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r;
}

enum { V_S = 0, V_SF = 1, V_L4 = 2 };

// signed int8x4 words for one qs entry (8 weights)
template <bool FAST>
static __device__ __forceinline__ void signed_entry(const uint32_t e, int & lo, int & hi) {
    const uint2 g = ((const uint2 *) d_grid)[e & 0x1FF];
    if constexpr (!FAST) {
        const uint32_t signs = unpack_ksigns(e >> 9);
        const int s0 = vcmpne4(signs & 0x08040201, 0); lo = vsub4(g.x ^ s0, s0);
        const int s1 = vcmpne4(signs & 0x80402010, 0); hi = vsub4(g.y ^ s1, s1);
    } else {
        const uint32_t s7 = e >> 9;
        const uint32_t s  = s7 | ((__popc(s7) & 1) << 7);
        const uint32_t b0 = mul_u24(s & 0xF, 0x00204081u) & 0x01010101u;
        const uint32_t b1 = mul_u24(s >> 4,  0x00204081u) & 0x01010101u;
        lo = (g.x ^ ((0x80808080u - b0) ^ 0x80808080u)) + b0;
        hi = (g.y ^ ((0x80808080u - b1) ^ 0x80808080u)) + b1;
    }
}

struct wsrc { const block_iq2_xs * blk; const int4 * q8w; const uint32_t * sc; };

// weights of sub-block s (0..7) of block ib: 8 signed int8x4 words + scale byte + d
template <int V>
static __device__ __forceinline__ void load_sub(const wsrc & W, const int ib, const int s, int w[8], int & sc, float & dw) {
    if constexpr (V == V_L4) {
        const int4 a = W.q8w[(size_t) ib*16 + 2*s + 0];
        const int4 b = W.q8w[(size_t) ib*16 + 2*s + 1];
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
        const uint32_t p = W.sc[(size_t) ib*8 + s];
        sc = (p >> 16) & 0xFF;
        dw = __half2float(__ushort_as_half((unsigned short) (p & 0xFFFF)));
    } else {
        const block_iq2_xs * bq = W.blk + ib;
        const int q0 = get_int_b2(bq->qs, 2*s + 0);
        const int q1 = get_int_b2(bq->qs, 2*s + 1);
        signed_entry<V == V_SF>(q0 & 0xFFFF, w[0], w[1]);
        signed_entry<V == V_SF>((uint32_t) q0 >> 16, w[2], w[3]);
        signed_entry<V == V_SF>(q1 & 0xFFFF, w[4], w[5]);
        signed_entry<V == V_SF>((uint32_t) q1 >> 16, w[6], w[7]);
        sc = bq->scales[s];
        dw = __half2float(__ushort_as_half(bq->d));
    }
}

static __device__ __forceinline__ int stock_sumi(const int sumi0, const int sumi1, const int sc) {
    return (sumi0*(sc & 0xF) + sumi1*(sc >> 4) + (sumi0 + sumi1)/2)/4;
}

// exhaustive dequant self-test: signed bytes of every qs value (9-bit grid index | 7-bit sign index), stock and fast sign
__global__ void dequant_selftest(uint2 * out_stock, uint2 * out_fast) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    int lo, hi;
    signed_entry<false>(e, lo, hi); out_stock[e] = make_uint2(lo, hi);
    signed_entry<true>(e, lo, hi);  out_fast[e]  = make_uint2(lo, hi);
}

// ---------------- batched GEMV (n = 1..8), mmvq-shaped ----------------
template <int V, int NC>
__launch_bounds__((NC <= 4 ? 2 : 1)*64, 1)
__global__ void gemv(const wsrc W, const block_q8_1 * __restrict__ y, float * __restrict__ dst, const int m, const int nb) {
    constexpr int NW  = NC <= 4 ? 2 : 1;
    constexpr int RPB = NC == 1 ? 1 : 2;
    const int tid = threadIdx.x;
    const int t   = tid % 8;
    const int row0 = blockIdx.x * RPB;
    const int nbk  = nb * 8;   // q8_1 blocks per column
    float acc[NC][RPB];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) { acc[j][i] = 0.0f; }
    }
    for (int kb = tid / 8; kb < nb; kb += NW*8) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) {
            int w[8]; int sc; float dw;
            load_sub<V>(W, (row0 + i)*nb + kb, t, w, sc, dw);
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                const block_q8_1 * by = y + (size_t) j*nbk + kb*8 + t;
                const int * u = (const int *) by->qs;
                int sumi0 = 0, sumi1 = 0;
                sumi0 = dp4a(w[0], u[0], sumi0); sumi0 = dp4a(w[1], u[1], sumi0);
                sumi0 = dp4a(w[2], u[2], sumi0); sumi0 = dp4a(w[3], u[3], sumi0);
                sumi1 = dp4a(w[4], u[4], sumi1); sumi1 = dp4a(w[5], u[5], sumi1);
                sumi1 = dp4a(w[6], u[6], sumi1); sumi1 = dp4a(w[7], u[7], sumi1);
                const float d = dw * __low2float(by->ds);
                acc[j][i] += d * stock_sumi(sumi0, sumi1, sc);
            }
        }
    }
    __shared__ float red[NW > 1 ? NC*RPB*64 : 1];
    if constexpr (NW > 1) {
        if (tid >= 64) {
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int i = 0; i < RPB; ++i) { red[(j*RPB + i)*64 + tid - 64] = acc[j][i]; }
            }
        }
        __syncthreads();
        if (tid >= 64) { return; }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) {
            float v = acc[j][i];
            if constexpr (NW > 1) { v += red[(j*RPB + i)*64 + tid]; }
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) { v += __shfl_xor(v, off, 64); }
            if (tid == 0) { dst[(size_t) j*m + row0 + i] = v; }
        }
    }
}

// ---------------- prefill GEMM (n multiple of 64) ----------------
#define GBM 64
#define GBN 64
#define GKS 4                         // sub-blocks per K step (128 weights)
#define GSTR (GKS*32 + 16)            // LDS row stride in bytes (padded)

template <int V>
__launch_bounds__(256, 2)
__global__ void gemm(const wsrc W, const block_q8_1 * __restrict__ y, float * __restrict__ dst, const int m, const int nb) {
    __shared__ __attribute__((aligned(16))) int8_t wq[GBM*GSTR];
    __shared__ uint32_t wsc[GBM*GKS];
    __shared__ __attribute__((aligned(16))) int8_t aq[GBN*GSTR];
    __shared__ float ad[GBN*GKS];
    const int tid = threadIdx.x;
    const int tr = tid / 16, tc = tid % 16;          // compute mapping: rows tr*4.., cols tc*4..
    const int row_base = blockIdx.x * GBM, col_base = blockIdx.y * GBN;
    const int nbk = nb * 8;
    const int lr = tid / GKS, ls = tid % GKS;        // loader mapping: one (row or col, sub-block) per thread
    float acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { acc[r][c] = 0.0f; }
    }
    for (int kb = 0; kb < nb; ++kb) {
        for (int half = 0; half < 2; ++half) {
            const int s = half*GKS + ls;
            {   // weights: row lr, sub-block s
                int w[8]; int sc; float dw;
                load_sub<V>(W, (row_base + lr)*nb + kb, s, w, sc, dw);
                int4 * dstw = (int4 *) (wq + lr*GSTR + ls*32);
                dstw[0] = make_int4(w[0], w[1], w[2], w[3]);
                dstw[1] = make_int4(w[4], w[5], w[6], w[7]);
                wsc[lr*GKS + ls] = (uint32_t) __half_as_ushort(__float2half(dw)) | ((uint32_t) sc << 16);
            }
            {   // activations: column lr, sub-block s
                const block_q8_1 * by = y + (size_t) (col_base + lr)*nbk + kb*8 + s;
                const int * u = (const int *) by->qs;
                int4 * dsta = (int4 *) (aq + lr*GSTR + ls*32);
                dsta[0] = make_int4(u[0], u[1], u[2], u[3]);
                dsta[1] = make_int4(u[4], u[5], u[6], u[7]);
                ad[lr*GKS + ls] = __low2float(by->ds);
            }
            __syncthreads();
#pragma unroll 1
            for (int ss = 0; ss < GKS; ++ss) {
                int a[4][8];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int4 * p = (const int4 *) (aq + (tc*4 + c)*GSTR + ss*32);
                    const int4 x0 = p[0], x1 = p[1];
                    a[c][0] = x0.x; a[c][1] = x0.y; a[c][2] = x0.z; a[c][3] = x0.w;
                    a[c][4] = x1.x; a[c][5] = x1.y; a[c][6] = x1.z; a[c][7] = x1.w;
                }
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int4 * p = (const int4 *) (wq + (tr*4 + r)*GSTR + ss*32);
                    const int4 x0 = p[0], x1 = p[1];
                    const uint32_t pk = wsc[(tr*4 + r)*GKS + ss];
                    const int sc = (pk >> 16) & 0xFF;
                    const float dw = __half2float(__ushort_as_half((unsigned short) (pk & 0xFFFF)));
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        int sumi0 = 0, sumi1 = 0;
                        sumi0 = dp4a(x0.x, a[c][0], sumi0); sumi0 = dp4a(x0.y, a[c][1], sumi0);
                        sumi0 = dp4a(x0.z, a[c][2], sumi0); sumi0 = dp4a(x0.w, a[c][3], sumi0);
                        sumi1 = dp4a(x1.x, a[c][4], sumi1); sumi1 = dp4a(x1.y, a[c][5], sumi1);
                        sumi1 = dp4a(x1.z, a[c][6], sumi1); sumi1 = dp4a(x1.w, a[c][7], sumi1);
                        const float d = dw * ad[(tc*4 + c)*GKS + ss];
                        acc[r][c] += d * stock_sumi(sumi0, sumi1, sc);
                    }
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { dst[(size_t) (col_base + tc*4 + c)*m + row_base + tr*4 + r] = acc[r][c]; }
    }
}

// ---------------- host ----------------
static float h2f(uint16_t h) { __half hh; memcpy(&hh, &h, 2); return __half2float(hh); }

struct tensor_t { std::string name; int m, k, nb; std::vector<block_iq2_xs> blk; };

static void dequant_row(const tensor_t & T, int r, std::vector<double> & w) {
    w.resize(T.k);
    for (int kb = 0; kb < T.nb; ++kb) {
        const block_iq2_xs & b = T.blk[(size_t) r*T.nb + kb];
        const double d = h2f(b.d);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const double db[2] = { d * (0.5 + (b.scales[ib32] & 0xF)) * 0.25, d * (0.5 + (b.scales[ib32] >> 4)) * 0.25 };
            for (int l = 0; l < 4; ++l) {
                const uint16_t e = b.qs[4*ib32 + l];
                const uint8_t * grid = (const uint8_t *) &h_iq2xs_grid[e & 511];
                const uint32_t s7 = e >> 9; const uint32_t signs = s7 ^ ((__builtin_popcount(s7) & 1) << 7);
                for (int j = 0; j < 8; ++j) { w[kb*256 + ib32*32 + l*8 + j] = db[l/2] * grid[j] * ((signs >> j) & 1 ? -1.0 : 1.0); }
            }
        }
    }
}

static std::vector<block_q8_1> make_q8_1(int k, int ncols, uint32_t seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<block_q8_1> y((size_t) ncols * (k/32));
    for (size_t b = 0; b < y.size(); ++b) {
        float x[32]; float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { x[i] = nd(rng); amax = std::max(amax, fabsf(x[i])); }
        const float d = amax / 127.0f; int sum = 0;
        for (int i = 0; i < 32; ++i) { const int q = (int) roundf(x[i] / d); y[b].qs[i] = (int8_t) q; sum += q; }
        y[b].ds = __floats2half2_rn(d, d * sum);
    }
    return y;
}

static double oracle_dot(const std::vector<double> & w, const block_q8_1 * yc, int k) {
    double acc = 0.0;
    for (int b = 0; b < k/32; ++b) {
        const double d8 = __half2float(__low2half(yc[b].ds));
        for (int i = 0; i < 32; ++i) { acc += w[b*32 + i] * d8 * yc[b].qs[i]; }
    }
    return acc;
}

template <class F>
static double time_it(F launch, int iters, std::vector<double> & reps) {
    hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    for (int i = 0; i < 5; ++i) { launch(); }
    HIP_CHECK(hipDeviceSynchronize());
    reps.clear();
    for (int r = 0; r < 7; ++r) {
        HIP_CHECK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i) { launch(); }
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); reps.push_back(1000.0 * ms / iters);
    }
    std::vector<double> s = reps; std::sort(s.begin(), s.end());
    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    return s[s.size()/2];
}

typedef void (*gemv_t)(const wsrc, const block_q8_1 *, float *, const int, const int);
typedef void (*gemm_t)(const wsrc, const block_q8_1 *, float *, const int, const int);

template <int V> static gemv_t pick_gemv(int nc) {
    switch (nc) {
        case 1: return gemv<V, 1>; case 2: return gemv<V, 2>; case 3: return gemv<V, 3>; case 4: return gemv<V, 4>;
        case 5: return gemv<V, 5>; case 6: return gemv<V, 6>; case 7: return gemv<V, 7>; default: return gemv<V, 8>;
    }
}

int main(int argc, char ** argv) {
    if (argc < 7 || (argc - 2) % 5 != 0) { fprintf(stderr, "usage: %s <gguf> {name m k abs_offset nbytes}...\n", argv[0]); return 1; }
    const int fd = open(argv[1], O_RDONLY); if (fd < 0) { perror("open"); return 1; }
    std::vector<tensor_t> T;
    for (int a = 2; a < argc; a += 5) {
        tensor_t t; t.name = argv[a]; t.m = atoi(argv[a+1]); t.k = atoi(argv[a+2]); t.nb = t.k / QK_K;
        const long long off = atoll(argv[a+3]); const size_t nbytes = (size_t) atoll(argv[a+4]);
        if (nbytes != (size_t) t.m * t.nb * sizeof(block_iq2_xs)) { fprintf(stderr, "size mismatch for %s\n", t.name.c_str()); return 1; }
        t.blk.resize((size_t) t.m * t.nb);
        if (pread(fd, t.blk.data(), nbytes, off) != (ssize_t) nbytes) { perror("pread"); return 1; }
        T.push_back(std::move(t));
    }
    close(fd);
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_grid), h_iq2xs_grid, sizeof(h_iq2xs_grid)));
    {   // host reference = exactly the L4 plane builder below
        uint2 * d0; uint2 * d1; std::vector<uint2> h0(65536), h1(65536);
        HIP_CHECK(hipMalloc(&d0, 65536*sizeof(uint2))); HIP_CHECK(hipMalloc(&d1, 65536*sizeof(uint2)));
        hipLaunchKernelGGL(dequant_selftest, dim3(256), dim3(256), 0, 0, d0, d1);
        HIP_CHECK(hipMemcpy(h0.data(), d0, 65536*sizeof(uint2), hipMemcpyDeviceToHost)); HIP_CHECK(hipMemcpy(h1.data(), d1, 65536*sizeof(uint2), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d0)); HIP_CHECK(hipFree(d1));
        int bad0 = 0, bad1 = 0;
        for (uint32_t e = 0; e < 65536; ++e) {
            int8_t ref[8];
            const uint8_t * grid = (const uint8_t *) &h_iq2xs_grid[e & 511];
            const uint32_t s7 = e >> 9; const uint32_t signs = s7 ^ ((__builtin_popcount(s7) & 1) << 7);
            for (int j = 0; j < 8; ++j) { ref[j] = (int8_t) ((signs >> j) & 1 ? -(int) grid[j] : (int) grid[j]); }
            bad0 += memcmp(ref, &h0[e], 8) != 0; bad1 += memcmp(ref, &h1[e], 8) != 0;
        }
        printf("dequant selftest over all 65536 qs values vs host L4 plane builder: stock-sign mismatches %d, fast-sign mismatches %d\n", bad0, bad1);
    }
    const char * vname[3] = {"S  stock 74B", "Sf stock 74B + fast sign", "L4 int8 plane + (d,sc) plane"};
    const int vbpb[3] = {74, 74, 288};
    const int NP = 512;   // prefill columns

    printf("tensor | m x k | kernel | n | variant | bytes/block | us (spread) | vs S | max|err|/rms(oracle) | rows*cols bitwise != S\n");
    for (const tensor_t & t : T) {
        const size_t nblk = (size_t) t.m * t.nb;
        // ---- L4 planes (host) ----
        std::vector<int8_t> q8p(nblk * 256); std::vector<uint32_t> scp(nblk * 8);
        for (size_t i = 0; i < nblk; ++i) {
            const block_iq2_xs & b = t.blk[i];
            for (int s = 0; s < 8; ++s) {
                scp[i*8 + s] = (uint32_t) b.d | ((uint32_t) b.scales[s] << 16);
                for (int l = 0; l < 4; ++l) {
                    const uint16_t e = b.qs[4*s + l];
                    const uint8_t * grid = (const uint8_t *) &h_iq2xs_grid[e & 511];
                    const uint32_t s7 = e >> 9; const uint32_t signs = s7 ^ ((__builtin_popcount(s7) & 1) << 7);
                    for (int j = 0; j < 8; ++j) { q8p[i*256 + s*32 + l*8 + j] = (int8_t) ((signs >> j) & 1 ? -(int) grid[j] : (int) grid[j]); }
                }
            }
        }
        block_iq2_xs * dB; int4 * dQ; uint32_t * dS;
        HIP_CHECK(hipMalloc(&dB, nblk*74)); HIP_CHECK(hipMemcpy(dB, t.blk.data(), nblk*74, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dQ, nblk*256)); HIP_CHECK(hipMemcpy(dQ, q8p.data(), nblk*256, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dS, nblk*32)); HIP_CHECK(hipMemcpy(dS, scp.data(), nblk*32, hipMemcpyHostToDevice));
        const wsrc W = {dB, dQ, dS};
        std::vector<block_q8_1> y = make_q8_1(t.k, NP, 4321);
        block_q8_1 * dy; float * dd;
        HIP_CHECK(hipMalloc(&dy, y.size()*sizeof(block_q8_1))); HIP_CHECK(hipMemcpy(dy, y.data(), y.size()*sizeof(block_q8_1), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dd, (size_t) t.m*NP*sizeof(float)));

        // ---- oracle: all rows x 8 columns (batched), 64 sampled rows x 512 columns (prefill) ----
        std::vector<double> ref8((size_t) t.m*8), refP((size_t) 64*NP); std::vector<double> wrow;
        for (int r = 0; r < t.m; ++r) {
            dequant_row(t, r, wrow);
            for (int j = 0; j < 8; ++j) { ref8[(size_t) j*t.m + r] = oracle_dot(wrow, &y[(size_t) j*(t.k/32)], t.k); }
            if (r % (t.m/64) == 0 && r / (t.m/64) < 64) {
                const int ri = r / (t.m/64);
                for (int j = 0; j < NP; ++j) { refP[(size_t) ri*NP + j] = oracle_dot(wrow, &y[(size_t) j*(t.k/32)], t.k); }
            }
        }
        double rms8 = 0; for (double v : ref8) { rms8 += v*v; } rms8 = sqrt(rms8 / ref8.size());
        double rmsP = 0; for (double v : refP) { rmsP += v*v; } rmsP = sqrt(rmsP / refP.size());

        // ---- batched GEMV ----
        for (int nc : {1, 2, 4, 8}) {
            std::vector<float> outS((size_t) t.m*nc), out((size_t) t.m*nc); double tS = 0;
            for (int v = 0; v < 3; ++v) {
                gemv_t k = v == 0 ? pick_gemv<V_S>(nc) : v == 1 ? pick_gemv<V_SF>(nc) : pick_gemv<V_L4>(nc);
                const int rpb = nc == 1 ? 1 : 2, thr = (nc <= 4 ? 2 : 1)*64;
                auto launch = [&] { hipLaunchKernelGGL(k, dim3(t.m/rpb), dim3(thr), 0, 0, W, dy, dd, t.m, t.nb); };
                std::vector<double> reps; const double us = time_it(launch, 100, reps);
                const double spread = 100.0 * (*std::max_element(reps.begin(), reps.end()) - *std::min_element(reps.begin(), reps.end())) / us;
                launch(); HIP_CHECK(hipMemcpy(out.data(), dd, out.size()*sizeof(float), hipMemcpyDeviceToHost));
                if (v == 0) { outS = out; tS = us; }
                double emax = 0; int ndiff = 0;
                for (int j = 0; j < nc; ++j) {
                    for (int r = 0; r < t.m; ++r) {
                        const size_t i = (size_t) j*t.m + r;
                        emax = std::max(emax, fabs(out[i] - ref8[i]));
                        ndiff += memcmp(&out[i], &outS[i], sizeof(float)) != 0;
                    }
                }
                printf("%s | %dx%d | gemv | %d | %s | %d | %.2f (%.1f%%) | %+.1f%% | %.2e | %d/%zu\n", t.name.c_str(), t.m, t.k, nc, vname[v], vbpb[v],
                       us, spread, 100.0*(us/tS - 1.0), emax/rms8, ndiff, out.size());
                fflush(stdout);
            }
        }
        // ---- prefill GEMM n=512 ----
        {
            std::vector<float> outS((size_t) t.m*NP), out((size_t) t.m*NP); double tS = 0;
            for (int v = 0; v < 3; ++v) {
                gemm_t k = v == 0 ? gemm<V_S> : v == 1 ? gemm<V_SF> : gemm<V_L4>;
                auto launch = [&] { hipLaunchKernelGGL(k, dim3(t.m/GBM, NP/GBN), dim3(256), 0, 0, W, dy, dd, t.m, t.nb); };
                std::vector<double> reps; const double us = time_it(launch, 10, reps);
                const double spread = 100.0 * (*std::max_element(reps.begin(), reps.end()) - *std::min_element(reps.begin(), reps.end())) / us;
                launch(); HIP_CHECK(hipMemcpy(out.data(), dd, out.size()*sizeof(float), hipMemcpyDeviceToHost));
                if (v == 0) { outS = out; tS = us; }
                double emax = 0; int ndiff = 0;
                for (int ri = 0; ri < 64; ++ri) {
                    const int r = ri * (t.m/64);
                    for (int j = 0; j < NP; ++j) { emax = std::max(emax, fabs(out[(size_t) j*t.m + r] - refP[(size_t) ri*NP + j])); }
                }
                for (size_t i = 0; i < out.size(); ++i) { ndiff += memcmp(&out[i], &outS[i], sizeof(float)) != 0; }
                const double tflops = 2.0 * t.m * (double) t.k * NP / (us * 1e-6) / 1e12;
                printf("%s | %dx%d | gemm | %d | %s | %d | %.2f (%.1f%%) [%.1f TFLOPS] | %+.1f%% | %.2e | %d/%zu\n", t.name.c_str(), t.m, t.k, NP, vname[v], vbpb[v],
                       us, spread, tflops, 100.0*(us/tS - 1.0), emax/rmsP, ndiff, out.size());
                fflush(stdout);
            }
        }
        HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dQ)); HIP_CHECK(hipFree(dS)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dd));
    }
    return 0;
}
