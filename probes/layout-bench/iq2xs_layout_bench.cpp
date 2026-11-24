// iq2xs_layout_bench.cpp — IQ2_XS weight-layout microbenchmark on gfx906 (MI50), spike step 5.
//
// Loads real IQ2_XS tensors from the Qwen3.8-27B GGUF once (pread at offsets given on the command line, see run.sh),
// builds repacked GPU copies, and times n=1 GEMV kernels shaped like llama.cpp's stock mmvq for IQ2_XS on GCN
// (128 threads = 2 waves per row, 8 threads per 256-weight block, 32 weights per thread per step, q8_1 activations).
//
// Layouts:
//   S   stock AoS block_iq2_xs (74 B: d | qs[32] | scales[8]); stock vec_dot_iq2_xs_q8_1 math (reference)
//   L1  planar: qs plane (64 B/block, 8-byte aligned per thread), scales plane (8 B/block), d plane (fp16/block). Same bytes as S.
//   L2  aligned AoS: block padded to 80 B (qs at 0, scales at 64, d at 72; 16-byte aligned blocks). +8% bytes.
//   L3  qs plane + fused-scale plane: per 32-weight sub-block one half2 = d*(0.5+ls_lo)/4, d*(0.5+ls_hi)/4 (no d plane, no
//       integer scale fixup). 96 B/block, +30% bytes, but only 2 weight loads per thread (qs uint2 + scales dword).
// Each layout: (a) loads-only kernel (same loads, no grid gather / sign / dot) and (b) full dp4a kernel.
// Correctness: every full kernel vs a double-precision CPU oracle (dequantize_row_iq2_xs semantics x the same q8_1 activations).
//
// Build: see build.sh (ROCm 6.2 clang, gfx906).   Run: see run.sh.

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
#include "ksigns64.h"

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

#define QK_K 256
struct block_iq2_xs { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; };      // 74 B, d is fp16 bits
static_assert(sizeof(block_iq2_xs) == 74, "block_iq2_xs size");
struct __attribute__((aligned(16))) block_iq2_xs_a80 { uint16_t qs[32]; uint8_t scales[8]; uint16_t d; uint8_t pad[6]; };
static_assert(sizeof(block_iq2_xs_a80) == 80, "a80 size");
struct block_q8_1 { __half2 ds; int8_t qs[32]; };                                           // 36 B, ds = (d, d*sum)
static_assert(sizeof(block_q8_1) == 36, "q8_1 size");

__device__ uint64_t d_grid[512];
__device__ uint64_t d_ksigns[128];   // llama.cpp ksigns64: per 7-bit sign index, 8 byte masks (0xFF = negative), parity bit included

// ---- helpers identical to llama.cpp's HIP versions (vendors/hip.h, vecdotq.cuh, common.cuh) ----
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
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}
static __device__ __forceinline__ int get_int_b4(const void * x, const int i32) { return ((const int *) x)[i32]; }

// Signed grid bytes for one qs entry e (9-bit grid index | 7-bit sign index), as two dwords (weights 0-3, 4-7).
// SIGN 0: stock (unpack_ksigns + vcmpne4 + xor + vsub4, as llama.cpp)
// SIGN 1: computed byte mask, no quarter-rate multiply: b = per-byte sign bit (0/1) spread with a 24-bit multiply,
//         m = 0x00/0xFF per byte via bytes01_to_mask(b), signed = (g ^ m) + b. Exact because grid bytes are 8/25/43 (nonzero, < 128): a negative byte
//         becomes (g ^ 0xFF) + 1 = 256 - g with no carry out of the byte.
// SIGN 2: ksigns64 byte-mask table (one 8-byte gather), signed = (g ^ m) + (m & 0x01010101).
// SIGN 3: stock math with its quarter-rate multiplies folded away: unpack_ksigns' byte broadcast (s * 0x01010101) as one
//         v_perm_b32, vcmpne4's (x >> 7) * 0xff via bytes01_to_mask.
static __device__ __forceinline__ uint32_t mul_u24(uint32_t a, uint32_t b) {   // force v_mul_u32_u24 (clang lowers __umul24 to v_mul_lo_u32 here)
    uint32_t r; asm("v_mul_u32_u24 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b)); return r;
}
// per-byte 0/1 -> 0x00/0xFF without a multiply: 0x80 - b is 0x7F or 0x80 per byte (no borrows), then ^ 0x80.
// ((b << 8) - b would be canonicalized by LLVM into b * 255 = quarter-rate v_mul_lo_u32.)
static __device__ __forceinline__ uint32_t bytes01_to_mask(uint32_t b) { return (0x80808080u - b) ^ 0x80808080u; }
static __device__ __forceinline__ unsigned int vcmpne4_nomul(unsigned int a, unsigned int b) {
    const unsigned int x = a ^ b;
    const unsigned int ne_low_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    return bytes01_to_mask((ne_low_7bits | (x & 0x80808080)) >> 7);
}
template <int SIGN>
static __device__ __forceinline__ void signed_grid(const uint32_t e, int & grid_l, int & grid_h) {
    const uint2 g = ((const uint2 *) d_grid)[e & 0x1FF];
    if constexpr (SIGN == 0) {
        const uint32_t signs = unpack_ksigns(e >> 9);
        const int signs0 = vcmpne4(signs & 0x08040201, 0);
        grid_l = vsub4(g.x ^ signs0, signs0);
        const int signs1 = vcmpne4(signs & 0x80402010, 0);
        grid_h = vsub4(g.y ^ signs1, signs1);
    } else if constexpr (SIGN == 1) {
        const uint32_t s7 = e >> 9;
        const uint32_t s  = s7 | ((__popc(s7) & 1) << 7);
        const uint32_t b0 = mul_u24(s & 0xF, 0x00204081u) & 0x01010101u;   // bit j of the nibble -> bit 0 of byte j
        const uint32_t b1 = mul_u24(s >> 4,  0x00204081u) & 0x01010101u;
        grid_l = (g.x ^ bytes01_to_mask(b0)) + b0;
        grid_h = (g.y ^ bytes01_to_mask(b1)) + b1;
    } else if constexpr (SIGN == 3) {
        const uint32_t s7 = e >> 9;
        const uint32_t s1 = s7 ^ ((__popc(s7) & 1) << 7);
        const uint32_t signs = __builtin_amdgcn_perm(s1, s1, 0u);            // broadcast byte 0 to all 4 bytes
        const int signs0 = vcmpne4_nomul(signs & 0x08040201, 0);
        grid_l = vsub4(g.x ^ signs0, signs0);
        const int signs1 = vcmpne4_nomul(signs & 0x80402010, 0);
        grid_h = vsub4(g.y ^ signs1, signs1);
    } else {
        const uint2 m = ((const uint2 *) d_ksigns)[e >> 9];
        grid_l = (g.x ^ m.x) + (m.x & 0x01010101u);
        grid_h = (g.y ^ m.y) + (m.y & 0x01010101u);
    }
}

// 4 grid entries (32 weights) x 32 activations -> sumi0 (entries 0,1), sumi1 (entries 2,3). Same order as stock.
template <int SIGN>
static __device__ __forceinline__ void dot32(const uint16_t * q2, const block_q8_1 * by, int & sumi0, int & sumi1) {
    sumi0 = 0; sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int grid_l, grid_h;
        signed_grid<SIGN>(q2[l0/2], grid_l, grid_h);
        const int u0 = get_int_b4(by->qs, l0 + 0);
        const int u1 = get_int_b4(by->qs, l0 + 1);
        if (l0 < 4) { sumi0 = dp4a(grid_l, u0, sumi0); sumi0 = dp4a(grid_h, u1, sumi0); }
        else        { sumi1 = dp4a(grid_l, u0, sumi1); sumi1 = dp4a(grid_h, u1, sumi1); }
    }
}

// exhaustive check of SIGN 1/2 against SIGN 0 over all 65536 qs values
__global__ void sign_selftest(unsigned int * mism) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    int l0, h0, l1, h1, l2, h2, l3, h3;
    signed_grid<0>(e, l0, h0); signed_grid<1>(e, l1, h1); signed_grid<2>(e, l2, h2); signed_grid<3>(e, l3, h3);
    if (l1 != l0 || h1 != h0) { atomicAdd(&mism[0], 1u); }
    if (l2 != l0 || h2 != h0) { atomicAdd(&mism[1], 1u); }
    if (l3 != l0 || h3 != h0) { atomicAdd(&mism[2], 1u); }
}
static __device__ __forceinline__ int act_xor(const block_q8_1 * by) {
    int a = 0;
#pragma unroll
    for (int l = 0; l < 8; ++l) { a ^= get_int_b4(by->qs, l); }
    return a;
}

enum { LAY_S = 0, LAY_L1 = 1, LAY_L2 = 2, LAY_L3 = 3 };

struct wptrs { const void * a; const void * b; const void * c; };

// one thread step: sub-block t (0..7) of weight block kb of row `row`; by = activation block for that sub-block
template <int LAY, bool LOADS_ONLY, int SIGN = 0>
static __device__ __forceinline__ float step(const wptrs w, const int nb, const int row, const int kb, const int t,
                                             const block_q8_1 * __restrict__ by) {
    const int ib = row*nb + kb; // 32-bit index math like stock mmvq (max 17408*68 blocks)
    if constexpr (LAY == LAY_S) {
        const block_iq2_xs * bq2 = (const block_iq2_xs *) w.a + ib;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, 2*t + 0), get_int_b2(bq2->qs, 2*t + 1));
        const int sc = bq2->scales[t];
        const float d = __half2float(__ushort_as_half(bq2->d)) * __low2float(by->ds);
        if constexpr (LOADS_ONLY) { return d * (q2_packed.x ^ q2_packed.y ^ sc ^ act_xor(by)); }
        int sumi0, sumi1; dot32<SIGN>((const uint16_t *) &q2_packed, by, sumi0, sumi1);
        const int sumi = (sumi0*(sc & 0xF) + sumi1*(sc >> 4) + (sumi0 + sumi1)/2)/4;
        return d * sumi;
    } else if constexpr (LAY == LAY_L1 || LAY == LAY_L2) {
        uint2 q2u; int sc; float dw;
        if constexpr (LAY == LAY_L1) {
            q2u = ((const uint2 *) w.a)[ib*8 + t];
            sc  = ((const uint8_t *) w.b)[ib*8 + t];
            dw  = __half2float(__ushort_as_half(((const uint16_t *) w.c)[ib]));
        } else {
            const block_iq2_xs_a80 * bq = (const block_iq2_xs_a80 *) w.a + ib;
            q2u = ((const uint2 *) bq->qs)[t];
            sc  = bq->scales[t];
            dw  = __half2float(__ushort_as_half(bq->d));
        }
        const float d = dw * __low2float(by->ds);
        if constexpr (LOADS_ONLY) { return d * (int(q2u.x ^ q2u.y) ^ sc ^ act_xor(by)); }
        int sumi0, sumi1; dot32<SIGN>((const uint16_t *) &q2u, by, sumi0, sumi1);
        const int sumi = (sumi0*(sc & 0xF) + sumi1*(sc >> 4) + (sumi0 + sumi1)/2)/4;
        return d * sumi;
    } else { // L3
        const uint2   q2u = ((const uint2 *) w.a)[ib*8 + t];
        const __half2 fs  = ((const __half2 *) w.b)[ib*8 + t];
        const float d8 = __low2float(by->ds);
        if constexpr (LOADS_ONLY) { return d8 * __low2float(fs) * (int(q2u.x ^ q2u.y) ^ act_xor(by)) + __high2float(fs); }
        int sumi0, sumi1; dot32<SIGN>((const uint16_t *) &q2u, by, sumi0, sumi1);
        return d8 * (__low2float(fs) * sumi0 + __high2float(fs) * sumi1);
    }
}

// n=1 GEMV, one row per 128-thread block (mirrors stock mmvq ncols_dst=1 on GCN: nwarps=2, rows_per_block=1, VDR 2)
template <int LAY, bool LOADS_ONLY, int SIGN = 0>
__launch_bounds__(128, 1)
__global__ void gemv(const wptrs w, const block_q8_1 * __restrict__ y, float * __restrict__ dst, const int nb) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int t   = tid % 8;
    float acc = 0.0f;
    for (int kb = tid / 8; kb < nb; kb += 16) {
        acc += step<LAY, LOADS_ONLY, SIGN>(w, nb, row, kb, t, y + kb*8 + t);
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) { acc += __shfl_xor(acc, off, 64); }
    __shared__ float red;
    if (tid == 64) { red = acc; }
    __syncthreads();
    if (tid == 0) { dst[row] = acc + red; }
}

// ---------------- host ----------------
static float h2f(uint16_t h) { __half hh; memcpy(&hh, &h, 2); return __half2float(hh); }
static uint16_t f2h(float f) { __half hh = __float2half(f); uint16_t h; memcpy(&h, &hh, 2); return h; }

struct tensor_t { std::string name; int m, k, nb; std::vector<block_iq2_xs> blk; };

static void cpu_oracle(const tensor_t & T, const std::vector<block_q8_1> & y, std::vector<double> & out) {
    out.assign(T.m, 0.0);
    for (int r = 0; r < T.m; ++r) {
        double acc = 0.0;
        for (int kb = 0; kb < T.nb; ++kb) {
            const block_iq2_xs & b = T.blk[(size_t) r*T.nb + kb];
            const double d = h2f(b.d);
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const block_q8_1 & q = y[kb*8 + ib32];
                const double d8 = __half2float(__low2half(q.ds));
                const double db[2] = { d * (0.5 + (b.scales[ib32] & 0xF)) * 0.25, d * (0.5 + (b.scales[ib32] >> 4)) * 0.25 };
                for (int l = 0; l < 4; ++l) {
                    const uint16_t e = b.qs[4*ib32 + l];
                    const uint8_t * grid = (const uint8_t *) &h_iq2xs_grid[e & 511];
                    uint32_t s7 = e >> 9; uint32_t signs = s7 ^ ((__builtin_popcount(s7) & 1) << 7);
                    for (int j = 0; j < 8; ++j) {
                        const double wv = db[l/2] * grid[j] * ((signs >> j) & 1 ? -1.0 : 1.0);
                        acc += wv * d8 * q.qs[8*l + j];
                    }
                }
            }
        }
        out[r] = acc;
    }
}

static std::vector<block_q8_1> make_q8_1(int k, uint32_t seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<block_q8_1> y(k/32);
    for (int b = 0; b < k/32; ++b) {
        float x[32]; float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { x[i] = nd(rng); amax = std::max(amax, fabsf(x[i])); }
        const float d = amax / 127.0f; int sum = 0;
        for (int i = 0; i < 32; ++i) { int q = (int) roundf(x[i] / d); y[b].qs[i] = (int8_t) q; sum += q; }
        y[b].ds = __floats2half2_rn(d, d * sum);
    }
    return y;
}

typedef void (*kern_t)(const wptrs, const block_q8_1 *, float *, const int);

static double time_kernel(kern_t k, const wptrs & w, const block_q8_1 * dy, float * dd, int m, int nb, std::vector<double> & reps) {
    hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    for (int i = 0; i < 20; ++i) { hipLaunchKernelGGL(k, dim3(m), dim3(128), 0, 0, w, dy, dd, nb); }
    HIP_CHECK(hipDeviceSynchronize());
    reps.clear();
    const int iters = 100;
    for (int r = 0; r < 7; ++r) {
        HIP_CHECK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i) { hipLaunchKernelGGL(k, dim3(m), dim3(128), 0, 0, w, dy, dd, nb); }
        HIP_CHECK(hipEventRecord(e1)); HIP_CHECK(hipEventSynchronize(e1));
        float ms; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1)); reps.push_back(1000.0 * ms / iters);
    }
    std::vector<double> s = reps; std::sort(s.begin(), s.end());
    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    return s[s.size()/2];
}

int main(int argc, char ** argv) {
    // usage: iq2xs_layout_bench <gguf> <name m k offset nbytes>...
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
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_ksigns), h_ksigns64, sizeof(h_ksigns64)));
    {
        unsigned int * dm; unsigned int hm[3];
        HIP_CHECK(hipMalloc(&dm, 3*sizeof(unsigned int))); HIP_CHECK(hipMemset(dm, 0, 3*sizeof(unsigned int)));
        hipLaunchKernelGGL(sign_selftest, dim3(65536/256), dim3(256), 0, 0, dm);
        HIP_CHECK(hipMemcpy(hm, dm, sizeof(hm), hipMemcpyDeviceToHost)); HIP_CHECK(hipFree(dm));
        printf("sign selftest over all 65536 qs values: SIGN1 (computed) mismatches %u, SIGN2 (ksigns64 table) mismatches %u, SIGN3 (stock, folded muls) mismatches %u\n", hm[0], hm[1], hm[2]);
    }

    printf("tensor | m x k | layout | bytes/block | loads-only us | floor GB/s (74B-equiv) | full us | eff GB/s (74B-equiv) | full vs S | max|err|/rms(oracle) | max|err-S|/rms | bitwise vs S\n");
    for (const tensor_t & t : T) {
        const size_t nblk = (size_t) t.m * t.nb;
        const double bytes74 = (double) nblk * 74.0;
        // --- repack on host ---
        std::vector<uint16_t> qsp(nblk * 32); std::vector<uint8_t> scp(nblk * 8); std::vector<uint16_t> dp(nblk);
        std::vector<block_iq2_xs_a80> a80(nblk); std::vector<__half2> fsp(nblk * 8);
        for (size_t i = 0; i < nblk; ++i) {
            const block_iq2_xs & b = t.blk[i];
            memcpy(&qsp[i*32], b.qs, 64); memcpy(&scp[i*8], b.scales, 8); dp[i] = b.d;
            memset(&a80[i], 0, sizeof(a80[i])); memcpy(a80[i].qs, b.qs, 64); memcpy(a80[i].scales, b.scales, 8); a80[i].d = b.d;
            const float d = h2f(b.d);
            for (int s = 0; s < 8; ++s) {
                fsp[i*8 + s] = __floats2half2_rn(d * (0.5f + (b.scales[s] & 0xF)) * 0.25f, d * (0.5f + (b.scales[s] >> 4)) * 0.25f);
            }
        }
        // --- device copies ---
        void *dS, *dQ, *dSc, *dD, *dA80, *dFs; float * dd; block_q8_1 * dy;
        HIP_CHECK(hipMalloc(&dS, nblk*74)); HIP_CHECK(hipMemcpy(dS, t.blk.data(), nblk*74, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dQ, nblk*64)); HIP_CHECK(hipMemcpy(dQ, qsp.data(), nblk*64, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dSc, nblk*8)); HIP_CHECK(hipMemcpy(dSc, scp.data(), nblk*8, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dD, nblk*2)); HIP_CHECK(hipMemcpy(dD, dp.data(), nblk*2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dA80, nblk*80)); HIP_CHECK(hipMemcpy(dA80, a80.data(), nblk*80, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dFs, nblk*32)); HIP_CHECK(hipMemcpy(dFs, fsp.data(), nblk*32, hipMemcpyHostToDevice));
        std::vector<block_q8_1> y = make_q8_1(t.k, 1234);
        HIP_CHECK(hipMalloc(&dy, y.size()*sizeof(block_q8_1))); HIP_CHECK(hipMemcpy(dy, y.data(), y.size()*sizeof(block_q8_1), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&dd, t.m*sizeof(float)));

        std::vector<double> ref; cpu_oracle(t, y, ref);
        double rms = 0; for (double v : ref) { rms += v*v; } rms = sqrt(rms / t.m);

        struct L { const char * name; int bpb; wptrs w; kern_t lo, full; };
        const L lays[] = {
            {"S  stock AoS 74B",        74, {dS, nullptr, nullptr},  gemv<LAY_S,  true>, gemv<LAY_S,  false>},
            {"L1 planar qs/sc/d",       74, {dQ, dSc, dD},           gemv<LAY_L1, true>, gemv<LAY_L1, false>},
            {"L2 aligned AoS 80B",      80, {dA80, nullptr, nullptr}, gemv<LAY_L2, true>, gemv<LAY_L2, false>},
            {"L3 qs + fused scale",     96, {dQ, dFs, nullptr},      gemv<LAY_L3, true>, gemv<LAY_L3, false>},
            {"Sc stock AoS, computed sign mask", 74, {dS, nullptr, nullptr},  nullptr, gemv<LAY_S,  false, 1>},
            {"L1c planar, computed sign mask",  74, {dQ, dSc, dD},            nullptr, gemv<LAY_L1, false, 1>},
            {"L2c aligned, computed sign mask", 80, {dA80, nullptr, nullptr}, nullptr, gemv<LAY_L2, false, 1>},
            {"L1t planar, ksigns64 table",      74, {dQ, dSc, dD},            nullptr, gemv<LAY_L1, false, 2>},
            {"L2t aligned, ksigns64 table",     80, {dA80, nullptr, nullptr}, nullptr, gemv<LAY_L2, false, 2>},
            {"L1f planar, stock signs folded",  74, {dQ, dSc, dD},            nullptr, gemv<LAY_L1, false, 3>},
            {"L2f aligned, stock signs folded", 80, {dA80, nullptr, nullptr}, nullptr, gemv<LAY_L2, false, 3>},
        };
        const int nlays = (int) (sizeof(lays) / sizeof(lays[0]));
        std::vector<float> outS(t.m), out(t.m);
        double tS = 0;
        for (int li = 0; li < nlays; ++li) {
            std::vector<double> reps;
            const double tlo = lays[li].lo ? time_kernel(lays[li].lo, lays[li].w, dy, dd, t.m, t.nb, reps) : NAN;
            const double tf  = time_kernel(lays[li].full, lays[li].w, dy, dd, t.m, t.nb, reps);
            const double spread = 100.0 * (*std::max_element(reps.begin(), reps.end()) - *std::min_element(reps.begin(), reps.end())) / tf;
            hipLaunchKernelGGL(lays[li].full, dim3(t.m), dim3(128), 0, 0, lays[li].w, dy, dd, t.nb);
            HIP_CHECK(hipMemcpy(out.data(), dd, t.m*sizeof(float), hipMemcpyDeviceToHost));
            if (li == 0) { outS = out; tS = tf; }
            double emax = 0, esmax = 0;
            for (int r = 0; r < t.m; ++r) { emax = std::max(emax, fabs(out[r] - ref[r])); esmax = std::max(esmax, (double) fabsf(out[r] - outS[r])); }
            int ndiff = 0; for (int r = 0; r < t.m; ++r) { ndiff += memcmp(&out[r], &outS[r], sizeof(float)) != 0; }
            printf("%s | %dx%d | %s | %d | %.2f | %.0f | %.2f (spread %.1f%%) | %.0f | %+.1f%% | %.2e | %.2e | rows bitwise != S: %d/%d\n",
                   t.name.c_str(), t.m, t.k, lays[li].name, lays[li].bpb, tlo, bytes74 / tlo / 1e3, tf, spread,
                   bytes74 / tf / 1e3, 100.0 * (tf / tS - 1.0), emax / rms, esmax / rms, ndiff, t.m);
            fflush(stdout);
        }
        HIP_CHECK(hipFree(dS)); HIP_CHECK(hipFree(dQ)); HIP_CHECK(hipFree(dSc)); HIP_CHECK(hipFree(dD));
        HIP_CHECK(hipFree(dA80)); HIP_CHECK(hipFree(dFs)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dd));
    }
    return 0;
}
