#pragma once
// TEMPORARY ablation probes for the IQ2_XS mmvq diagnosis (spike step 1). Not for production; results are wrong for PROBE != 0.
// IQ2XS_PROBE: 0 stock math (copy of vec_dot_iq2_xs_q8_1)
//              1 loads only: weight qs/scales/d + activation qs/ds loaded, no grid gather, no dp4a
//              2 no activation loads: q8_1 qs/ds replaced by values derived from the weights
//              3 no grid gather: grid values computed from the index instead of the iq2xs_grid LUT
//              4 wide weight loads: qs read as one 8-byte load instead of 4x 2-byte loads
//              5 no sign unpack: ksigns table/vcmpne/xor/sub skipped
#include "vecdotq.cuh"

//#define IQ2XS_PROBE_ROWS 1 // rows-per-block override at n=1 on GCN (probe)
#ifndef IQ2XS_PROBE
#define IQ2XS_PROBE 0
#endif

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_probe(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

#if IQ2XS_PROBE == 4
    const int2 q2_packed = *((const int2 *) (bq2->qs + 2*iqs)); // may be only 2-byte aligned; gfx9 runs in unaligned mode
#else
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
#endif
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

#if IQ2XS_PROBE == 1
    int acc = q2_packed.x ^ q2_packed.y ^ ls0 ^ ls1;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        acc ^= get_int_b4(bq8_1[iqs/2].qs, l);
    }
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * acc;
#else
    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
#if IQ2XS_PROBE == 3
        const uint32_t ix = q2[l0/2] & 0x1FF;
        const uint2 grid_pos = make_uint2(ix * 0x01010101u, (ix ^ 0x155) * 0x01010101u);
#else
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
#endif
#if IQ2XS_PROBE == 5
        const int grid_l = grid_pos.x;
        const int grid_h = grid_pos.y;
#else
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
#endif
#if IQ2XS_PROBE == 2
        const int u0 = q2_packed.x + l0;
        const int u1 = q2_packed.y + l0;
#else
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
#endif
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
#if IQ2XS_PROBE == 2
    const float d = __half2float(bq2->d);
#else
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
#endif
    return d * sumi;
#endif
}
