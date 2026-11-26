#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

// p104 k1: MMVQ variant with vdr 8, iqs in 0, 8, 16, 24 selects sub-block pair iqs/8 (32 contiguous quant bytes).
// The min term uses an exact dp4a sum of the q8_1 values, as vec_dot_q4_K_q8_1 does.
template <bool exact_min = true>
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_wide(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;
    const int j = iqs / 8;

    // 16-byte loads: the host asserts a 16-byte aligned matrix (block size 144 keeps the base alignment).
    const int4 * q4 = (const int4 *) (bq4_K->qs + 32*j);
    const int4 v0 = __ldg(q4 + 0);
    const int4 v1 = __ldg(q4 + 1);
    const int v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int jm = j & 1;

    const uint32_t s0 = __ldg(scales + jm + 0);
    const uint32_t s2 = __ldg(scales + jm + 2);
    const uint32_t s4 = __ldg(scales + jm + 4);

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + 2*j + i;
        const int * q8 = (const int *) bq8i->qs;

        int sumi_d = 0;
        int sumi_m = 0;
#pragma unroll
        for (int l = 0; l < QI8_1; ++l) {
            const int u = __ldg(q8 + l);
            sumi_d = ggml_cuda_dp4a((v[l] >> (4*i)) & 0x0F0F0F0F, u, sumi_d);
            if constexpr (exact_min) {
                sumi_m = ggml_cuda_dp4a(0x01010101, u, sumi_m);
            }
        }

        const float2 ds8f = __half22float2(__ldg(&bq8i->ds));
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        if constexpr (exact_min) {
            sumf_m += ds8f.x * (sumi_m * m[i]);
        } else {
            sumf_m += ds8f.y * m[i]; // q8_1 block sum as stored by quantize_q8_1 (as the MMQ path does), halves the dp4a count
        }
    }

    const float2 dm4f = __half22float2(__ldg(&bq4_K->dm));

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// p104 k1: Q5_K variant of vec_dot_q4_K_q8_1_wide, the 5th bit of the 32 values comes from all 32 qh bytes.
template <bool exact_min = true>
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_wide(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;
    const int j = iqs / 8;

    // 16-byte loads: the host asserts a 16-byte aligned matrix (block size 176 keeps the base alignment).
    const int4 * q5 = (const int4 *) (bq5_K->qs + 32*j);
    const int4 * qh = (const int4 *) bq5_K->qh;
    const int4 v0 = __ldg(q5 + 0);
    const int4 v1 = __ldg(q5 + 1);
    const int4 h0 = __ldg(qh + 0);
    const int4 h1 = __ldg(qh + 1);
    const int v[8]  = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    const int vh[8] = {h0.x >> (2*j), h0.y >> (2*j), h0.z >> (2*j), h0.w >> (2*j),
                       h1.x >> (2*j), h1.y >> (2*j), h1.z >> (2*j), h1.w >> (2*j)};

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int jm = j & 1;

    const uint32_t s0 = __ldg(scales + jm + 0);
    const uint32_t s2 = __ldg(scales + jm + 2);
    const uint32_t s4 = __ldg(scales + jm + 4);

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + 2*j + i;
        const int * q8 = (const int *) bq8i->qs;

        int sumi_d = 0;
        int sumi_m = 0;
#pragma unroll
        for (int l = 0; l < QI8_1; ++l) {
            const int u = __ldg(q8 + l);
            const int vl = ((v[l] >> (4*i)) & 0x0F0F0F0F) | ((vh[l] << (4 - i)) & 0x10101010);
            sumi_d = ggml_cuda_dp4a(vl, u, sumi_d);
            if constexpr (exact_min) {
                sumi_m = ggml_cuda_dp4a(0x01010101, u, sumi_m);
            }
        }

        const float2 ds8f = __half22float2(__ldg(&bq8i->ds));
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        if constexpr (exact_min) {
            sumf_m += ds8f.x * (sumi_m * m[i]);
        } else {
            sumf_m += ds8f.y * m[i];
        }
    }

    const float2 dm5f = __half22float2(__ldg(&bq5_K->dm));

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

#ifndef P104_Q6_VDR
#define P104_Q6_VDR 4
#endif
#ifndef P104_Q6_LD
#define P104_Q6_LD 1
#endif
// p104 k3: 4 contiguous ints from a 2-byte aligned address, as 16-bit loads (LD 0) or as 5 aligned 32-bit loads and
// funnel shifts (LD 1). Q6_K blocks are 210 bytes, so ql/qh are only 2-byte aligned in odd blocks.
template <int n>
static __device__ __forceinline__ void p104_load_ints_b2(const uint8_t * p, int * out) {
    const uintptr_t a = (uintptr_t) p;
    if constexpr (P104_Q6_LD == 2 && n == 4) {
        // two aligned 16-byte loads (the second only if the range crosses into it) and register selects
        const int4 * pa  = (const int4 *) (a & ~(uintptr_t) 15);
        const int    off = a & 15;
        const int4 A = __ldg(pa);
        const int4 B = off != 0 ? __ldg(pa + 1) : A;
        const int  sh = (off & 3) * 8;
        int w[5];
        switch (off >> 2) {
            case 0:  w[0] = A.x; w[1] = A.y; w[2] = A.z; w[3] = A.w; w[4] = B.x; break;
            case 1:  w[0] = A.y; w[1] = A.z; w[2] = A.w; w[3] = B.x; w[4] = B.y; break;
            case 2:  w[0] = A.z; w[1] = A.w; w[2] = B.x; w[3] = B.y; w[4] = B.z; break;
            default: w[0] = A.w; w[1] = B.x; w[2] = B.y; w[3] = B.z; w[4] = B.w; break;
        }
#pragma unroll
        for (int i = 0; i < n; ++i) {
            out[i] = __funnelshift_r(w[i], w[i+1], sh);
        }
    } else if constexpr (P104_Q6_LD >= 1) {
        const int * pa = (const int *) (a & ~(uintptr_t) 3);
        const int   sh = (a & 3) * 8;
        int w[n+1];
#pragma unroll
        for (int i = 0; i <= n; ++i) {
            w[i] = __ldg(pa + i);
        }
#pragma unroll
        for (int i = 0; i < n; ++i) {
            out[i] = __funnelshift_r(w[i], w[i+1], sh);
        }
    } else {
        const uint16_t * p16 = (const uint16_t *) p;
#pragma unroll
        for (int i = 0; i < n; ++i) {
            out[i] = __ldg(p16 + 2*i) | (__ldg(p16 + 2*i + 1) << 16);
        }
    }
}

// p104 k4: magic-number int->float (0x4B400000 = 12582912.0f) eliminates quarter-rate I2F on sm_61.
#define P104_MMVQ_MAGIC_I 0x4B400000
#define P104_MMVQ_MAGIC_F 12582912.0f

// p104 k3: upstream vec_dot_q6_K_q8_1 with the number of contiguous ints per thread as a template parameter. The
// upstream index math already keeps vdr = 4 inside one 16-value scale group, one q8_1 block and one qh shift.
template <int vdr, bool presub = false>
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_v(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset   = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift     = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    int vl[vdr];
    int vh[vdr];
    p104_load_ints_b2<vdr>(bq6_K->ql + 4*iqs, vl);
    p104_load_ints_b2<vdr>(bq6_K->qh + 4*((QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)), vh);

    const int8_t * scales = bq6_K->scales + scale_offset;

    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + 2*i;
        const int * q8 = (const int *) bq8i->qs + iqs % QI8_1;
        const int sc = __ldg(scales + 4*i);

        int sumi = P104_MMVQ_MAGIC_I;
#pragma unroll
        for (int j = 0; j < vdr; ++j) {
            const int vil = (vl[j] >> (4*i)) & 0x0F0F0F0F;
            const int vih = (((vh[j] >> vh_shift) >> (4*i)) << 4) & 0x30303030;
            const int u   = __ldg(q8 + j);
            if constexpr (presub) {
                // q - 32 per byte as int8 without a second dp4a (worth it once several columns share the weights):
                // x = q ^ 0x20 is q - 32 for q >= 32; for q < 32 bit 5 of x is set and bits 6, 7 must be set too
                const int x = (vil | vih) ^ 0x20202020;
                const int m = x & 0x20202020;
                sumi = ggml_cuda_dp4a(x | (m << 1) | (m << 2), u, sumi);
            } else {
                sumi = ggml_cuda_dp4a(vil | vih, u, sumi);
                sumi = ggml_cuda_dp4a(0xE0E0E0E0, u, sumi);
            }
        }
        const float fsum = __int_as_float(sumi) - P104_MMVQ_MAGIC_F;
        const float fsc  = __int_as_float(P104_MMVQ_MAGIC_I + sc) - P104_MMVQ_MAGIC_F;
        sumf = fmaf(__low2float(__ldg(&bq8i->ds)) * fsc, fsum, sumf);
    }

    return __half2float(__ldg(&bq6_K->d)) * sumf;
}

// p104 k1: on Pascal (sm_61) the K-quant mat-vec is latency bound with 8 weight bytes per thread.
// Wide variant: each thread handles a whole 32-byte quant run (two 16-byte loads) of one sub-block pair.
#ifndef P104_WIDE_MAX_NCOLS
#define P104_WIDE_MAX_NCOLS 3
#endif
#ifndef P104_NW
#define P104_NW 2
#endif
#ifndef P104_RPB
#define P104_RPB 4
#endif
#ifndef P104_NW_Q6
#define P104_NW_Q6 4
#endif
#ifndef P104_RPB_Q6
#define P104_RPB_Q6 4
#endif
// p104 k3: launch shape and Q6_K vdr for 2..3 columns (batched decode, -np 2)
#ifndef P104_NW_N23
#define P104_NW_N23 2
#endif
#ifndef P104_RPB_N23
#define P104_RPB_N23 2
#endif
#ifndef P104_NW_Q6_N23
#define P104_NW_Q6_N23 4
#endif
#ifndef P104_RPB_Q6_N23
#define P104_RPB_Q6_N23 4
#endif
#ifndef P104_Q6_VDR_N23
#define P104_Q6_VDR_N23 2
#endif
// p104 k3: launch shape for the wide Q4_K/Q5_K path at 4..8 columns (used only if P104_WIDE_MAX_NCOLS >= 4)
// p104 k3: from this many columns on, the wide vec_dots take the Q4_K/Q5_K min term from the stored q8_1 block sums
// (ds.y, like MMQ) instead of an exact dp4a sum: halves the dp4a work, changes rounding (KLD-gated). 99 = off.
// p104 k3: from this many columns on, Q6_K subtracts the 32 offset in the weights instead of with a second dp4a. 99 = off.
#ifndef P104_Q6_PRESUB_MINCOLS
#define P104_Q6_PRESUB_MINCOLS 2
#endif
#ifndef P104_DS_MINCOLS
#define P104_DS_MINCOLS 99
#endif
#ifndef P104_NW_N48
#define P104_NW_N48 2
#endif
#ifndef P104_RPB_N48
#define P104_RPB_N48 1
#endif
static constexpr __host__ __device__ bool mmvq_pascal_wide(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
}

// Types that get the Pascal nwarps / rows per block below (Q6_K keeps its vec_dot, only more rows per block).
static constexpr __host__ __device__ bool mmvq_pascal_tuned(ggml_type type) {
    return mmvq_pascal_wide(type) || type == GGML_TYPE_Q6_K;
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA1,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10,
    MMVQ_PARAMETERS_ORIN,
    MMVQ_PARAMETERS_PASCAL
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA1)
    return MMVQ_PARAMETERS_RDNA1;
#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_GB10
    return MMVQ_PARAMETERS_GB10;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_ORIN
    return MMVQ_PARAMETERS_ORIN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A && __CUDA_ARCH__ < GGML_CUDA_CC_VOLTA
    return MMVQ_PARAMETERS_PASCAL;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA1(cc)) {
        return MMVQ_PARAMETERS_RDNA1;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_GB10) {
        return MMVQ_PARAMETERS_GB10;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_ORIN) {
        return MMVQ_PARAMETERS_ORIN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_VOLTA) {
        return MMVQ_PARAMETERS_PASCAL;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        cc = ggml_cuda_highest_compiled_arch(cc);
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_GB10) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        switch (type) { // tuned on Radeon 8060S (gfx1151)
            case GGML_TYPE_Q8_0:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

// The wide vec_dot is used for up to P104_WIDE_MAX_NCOLS columns; with 4..8 columns it was slower than vdr 2.
static constexpr __host__ __device__ int get_vdr_mmvq_tbl(ggml_type type, mmvq_parameter_table_id table_id, int ncols_dst) {
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst <= 3 && type == GGML_TYPE_Q6_K) {
        return ncols_dst == 1 ? P104_Q6_VDR : P104_Q6_VDR_N23;
    }
    return table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst <= P104_WIDE_MAX_NCOLS && mmvq_pascal_wide(type) ? 8 : get_vdr_mmvq(type);
}

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda_tbl(ggml_type type, mmvq_parameter_table_id table_id, int ncols_dst) {
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst <= P104_WIDE_MAX_NCOLS) {
        switch (type) {
            case GGML_TYPE_Q4_K: return ncols_dst >= P104_DS_MINCOLS ? vec_dot_q4_K_q8_1_wide<false> : vec_dot_q4_K_q8_1_wide<true>;
            case GGML_TYPE_Q5_K: return ncols_dst >= P104_DS_MINCOLS ? vec_dot_q5_K_q8_1_wide<false> : vec_dot_q5_K_q8_1_wide<true>;
            default: break;
        }
    }
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst == 1 && type == GGML_TYPE_Q6_K && (P104_Q6_VDR != 2 || P104_Q6_LD != 0)) {
        return vec_dot_q6_K_q8_1_v<P104_Q6_VDR>;
    }
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst >= P104_Q6_PRESUB_MINCOLS && type == GGML_TYPE_Q6_K) {
        return ncols_dst <= 3 ? vec_dot_q6_K_q8_1_v<P104_Q6_VDR_N23, true> : vec_dot_q6_K_q8_1_v<2, true>;
    }
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst >= 2 && ncols_dst <= 3 && type == GGML_TYPE_Q6_K && (P104_Q6_VDR_N23 != 2 || P104_Q6_LD != 0)) {
        return vec_dot_q6_K_q8_1_v<P104_Q6_VDR_N23>;
    }
    return get_vec_dot_q_cuda(type);
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA1) {
        switch (ncols_dst) {
            case 1:
                return 1;
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    if (table_id == MMVQ_PARAMETERS_ORIN) {
        return calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
    }
    if (table_id == MMVQ_PARAMETERS_PASCAL) {
        if (ncols_dst == 1 && mmvq_pascal_tuned(type)) {
            return type == GGML_TYPE_Q6_K ? P104_NW_Q6 : P104_NW;
        }
        if (ncols_dst >= 2 && ncols_dst <= 3 && mmvq_pascal_tuned(type)) {
            return type == GGML_TYPE_Q6_K ? P104_NW_Q6_N23 : P104_NW_N23;
        }
        if (ncols_dst >= 4 && ncols_dst <= P104_WIDE_MAX_NCOLS && mmvq_pascal_wide(type)) {
            return P104_NW_N48;
        }
        return calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst == 1 && !small_k && mmvq_pascal_tuned(type)) {
        return type == GGML_TYPE_Q6_K ? P104_RPB_Q6 : P104_RPB;
    }
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst >= 2 && ncols_dst <= 3 && mmvq_pascal_tuned(type)) {
        return type == GGML_TYPE_Q6_K ? P104_RPB_Q6_N23 : P104_RPB_N23;
    }
    if (table_id == MMVQ_PARAMETERS_PASCAL && ncols_dst >= 4 && ncols_dst <= P104_WIDE_MAX_NCOLS && mmvq_pascal_wide(type)) {
        return P104_RPB_N48;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10 || table_id == MMVQ_PARAMETERS_RDNA1 || table_id == MMVQ_PARAMETERS_PASCAL) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_RDNA1 || table_id == MMVQ_PARAMETERS_PASCAL) && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_ORIN) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                // One row per warp (more rows per block, more loads in flight) for these types, measured on Jetson Orin Nano.
                // Other types are faster in isolated mat-vec too, but gained at most 2 % end to end.
                switch (type) {
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ4_NL:
                        return nwarps;
                    default:
                        break;
                }
                return mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// p104 aveA: register-light column-outer fused k-loop for Q4_K on Pascal (ncols_dst 2..3, no fusion).
// Loads rows_per_cuda_block weight rows once, then iterates columns loading each q8_1 block ONCE and sharing
// sumi_m across all rows in the block without keeping d8/fsum_m arrays live across rows.
template <int ncols_dst, int rows_per_cuda_block>
static __device__ __forceinline__ void p104_mmvq_kloop_q4_col(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int tid, const int blocks_per_row_x, const int blocks_per_iter,
        const int kbx_offset, const int stride_row_x, const uint32_t stride_col_y,
        float (&tmp)[ncols_dst][rows_per_cuda_block]) {
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qk;
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
    constexpr int vdr = 8;
    const int kqs = vdr * (tid % (qi / vdr));
    const int j_sub = kqs / 8;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);

        int   v[rows_per_cuda_block][8];
        float wd[rows_per_cuda_block][2];
        float wm[rows_per_cuda_block][2];

#pragma unroll
        for (int r = 0; r < rows_per_cuda_block; ++r) {
            const int row_kbx = kbx_offset + r * stride_row_x + kbx;
            const block_q4_K * bq = (const block_q4_K *) vx + row_kbx;
            const int4 * q4 = (const int4 *) (bq->qs + 32 * j_sub);
            const int4 v0 = __ldg(q4 + 0);
            const int4 v1 = __ldg(q4 + 1);
            v[r][0] = v0.x; v[r][1] = v0.y; v[r][2] = v0.z; v[r][3] = v0.w;
            v[r][4] = v1.x; v[r][5] = v1.y; v[r][6] = v1.z; v[r][7] = v1.w;

            const uint16_t * scales = (const uint16_t *) bq->scales;
            const int jm = j_sub & 1;
            const uint32_t s0 = __ldg(scales + jm + 0);
            const uint32_t s2 = __ldg(scales + jm + 2);
            const uint32_t s4 = __ldg(scales + jm + 4);
            const uint32_t hi = (uint32_t) -(int32_t) (j_sub >= 2);
            const uint32_t a0 = ((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi);
            const uint32_t a1 = ((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi);
            const float2 dmf = __half22float2(__ldg(&bq->dm));
            wd[r][0] = dmf.x * (__int_as_float(P104_MMVQ_MAGIC_I | (a0 & 0xff)) - P104_MMVQ_MAGIC_F);
            wd[r][1] = dmf.x * (__int_as_float(P104_MMVQ_MAGIC_I | ((a0 >> 8) & 0xff)) - P104_MMVQ_MAGIC_F);
            wm[r][0] = dmf.y * (__int_as_float(P104_MMVQ_MAGIC_I | (a1 & 0xff)) - P104_MMVQ_MAGIC_F);
            wm[r][1] = dmf.y * (__int_as_float(P104_MMVQ_MAGIC_I | ((a1 >> 8) & 0xff)) - P104_MMVQ_MAGIC_F);
        }

#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
#pragma unroll
            for (int i_sub = 0; i_sub < 2; ++i_sub) {
                const block_q8_1 * bq8i = &y[c * stride_col_y + kby + 2 * j_sub + i_sub];
                const int * q8 = (const int *) bq8i->qs;
                int sumi_m = P104_MMVQ_MAGIC_I;
                int sumi_d[rows_per_cuda_block];
#pragma unroll
                for (int r = 0; r < rows_per_cuda_block; ++r) {
                    sumi_d[r] = P104_MMVQ_MAGIC_I;
                }
#pragma unroll
                for (int l = 0; l < QI8_1; ++l) {
                    const int u = __ldg(q8 + l);
                    sumi_m = ggml_cuda_dp4a(0x01010101, u, sumi_m);
#pragma unroll
                    for (int r = 0; r < rows_per_cuda_block; ++r) {
                        sumi_d[r] = ggml_cuda_dp4a((v[r][l] >> (4 * i_sub)) & 0x0F0F0F0F, u, sumi_d[r]);
                    }
                }
                const float d8v = __low2float(__ldg(&bq8i->ds));
                const float fm  = d8v * (__int_as_float(sumi_m) - P104_MMVQ_MAGIC_F);
#pragma unroll
                for (int r = 0; r < rows_per_cuda_block; ++r) {
                    const float fd = __int_as_float(sumi_d[r]) - P104_MMVQ_MAGIC_F;
                    tmp[c][r] = fmaf(fd, wd[r][i_sub] * d8v, fmaf(-wm[r][i_sub], fm, tmp[c][r]));
                }
            }
        }
    }
}

// p104 v15: single-column hoisted-y + magic-float k-loop for Q4_K/Q5_K on Pascal (ncols_dst == 1, no fusion).
// Loads q8_1 u[2][8] once via int4 __ldg, computes exact sumi_m once across rows_per_cuda_block, and uses magic-float.
template <ggml_type type, int rows_per_cuda_block>
static __device__ __forceinline__ void p104_mmvq_kloop_q45_n1(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int tid, const int blocks_per_row_x, const int blocks_per_iter,
        const int kbx_offset, const int stride_row_x,
        float (&tmp)[1][rows_per_cuda_block]) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = 8;
    const int kqs = vdr * (tid % (qi / vdr));
    const int j_sub = kqs / 8;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);

        int   u[2][QI8_1];
        float d8[2];
        float fsum_m[2];
#pragma unroll
        for (int i_sub = 0; i_sub < 2; ++i_sub) {
            const block_q8_1 * bq8i = &y[kby + 2 * j_sub + i_sub];
            const int * q8 = (const int *) bq8i->qs;
            int sumi_m = P104_MMVQ_MAGIC_I;
#pragma unroll
            for (int l = 0; l < QI8_1; ++l) {
                const int uv = __ldg(q8 + l);
                u[i_sub][l] = uv;
                sumi_m = ggml_cuda_dp4a(0x01010101, uv, sumi_m);
            }
            const float d8v = __low2float(__ldg(&bq8i->ds));
            d8[i_sub] = d8v;
            fsum_m[i_sub] = d8v * (__int_as_float(sumi_m) - P104_MMVQ_MAGIC_F);
        }

#pragma unroll
        for (int r = 0; r < rows_per_cuda_block; ++r) {
            const int row_kbx = kbx_offset + r * stride_row_x + kbx;
            int v[8];
            int vh[8];
            const uint16_t * scales;
            float2 dmf;
            if constexpr (type == GGML_TYPE_Q4_K) {
                const block_q4_K * bq = (const block_q4_K *) vx + row_kbx;
                const int4 * q4 = (const int4 *) (bq->qs + 32 * j_sub);
                const int4 v0 = __ldg(q4 + 0);
                const int4 v1 = __ldg(q4 + 1);
                v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
                v[4] = v1.x; v[5] = v1.y; v[6] = v1.z; v[7] = v1.w;
                scales = (const uint16_t *) bq->scales;
                dmf = __half22float2(__ldg(&bq->dm));
            } else {
                const block_q5_K * bq = (const block_q5_K *) vx + row_kbx;
                const int4 * q5 = (const int4 *) (bq->qs + 32 * j_sub);
                const int4 * qh = (const int4 *) bq->qh;
                const int4 v0 = __ldg(q5 + 0);
                const int4 v1 = __ldg(q5 + 1);
                const int4 h0 = __ldg(qh + 0);
                const int4 h1 = __ldg(qh + 1);
                v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
                v[4] = v1.x; v[5] = v1.y; v[6] = v1.z; v[7] = v1.w;
                vh[0] = h0.x >> (2 * j_sub); vh[1] = h0.y >> (2 * j_sub);
                vh[2] = h0.z >> (2 * j_sub); vh[3] = h0.w >> (2 * j_sub);
                vh[4] = h1.x >> (2 * j_sub); vh[5] = h1.y >> (2 * j_sub);
                vh[6] = h1.z >> (2 * j_sub); vh[7] = h1.w >> (2 * j_sub);
                scales = (const uint16_t *) bq->scales;
                dmf = __half22float2(__ldg(&bq->dm));
            }
            const int jm = j_sub & 1;
            const uint32_t s0 = __ldg(scales + jm + 0);
            const uint32_t s2 = __ldg(scales + jm + 2);
            const uint32_t s4 = __ldg(scales + jm + 4);
            const uint32_t hi = (uint32_t) -(int32_t) (j_sub >= 2);
            const uint32_t a0 = ((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi);
            const uint32_t a1 = ((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi);
            const float wd[2] = {
                dmf.x * (__int_as_float(P104_MMVQ_MAGIC_I | (a0 & 0xff)) - P104_MMVQ_MAGIC_F),
                dmf.x * (__int_as_float(P104_MMVQ_MAGIC_I | ((a0 >> 8) & 0xff)) - P104_MMVQ_MAGIC_F)
            };
            const float wm[2] = {
                dmf.y * (__int_as_float(P104_MMVQ_MAGIC_I | (a1 & 0xff)) - P104_MMVQ_MAGIC_F),
                dmf.y * (__int_as_float(P104_MMVQ_MAGIC_I | ((a1 >> 8) & 0xff)) - P104_MMVQ_MAGIC_F)
            };

            float row_sum = 0.0f;
#pragma unroll
            for (int i_sub = 0; i_sub < 2; ++i_sub) {
                int sumi_d = P104_MMVQ_MAGIC_I;
#pragma unroll
                for (int l = 0; l < QI8_1; ++l) {
                    int vl = (v[l] >> (4 * i_sub)) & 0x0F0F0F0F;
                    if constexpr (type == GGML_TYPE_Q5_K) {
                        vl |= (vh[l] << (4 - i_sub)) & 0x10101010;
                    }
                    sumi_d = ggml_cuda_dp4a(vl, u[i_sub][l], sumi_d);
                }
                const float fd = __int_as_float(sumi_d) - P104_MMVQ_MAGIC_F;
                row_sum = fmaf(fd, wd[i_sub] * d8[i_sub], row_sum);
                row_sum = fmaf(-wm[i_sub], fsum_m[i_sub], row_sum);
            }
            tmp[0][r] += row_sum;
        }
    }
}

// p104 v15: hoisted-y + single -32*sum(u) k-loop for Q6_K on Pascal (ncols_dst 1..3, no fusion).
// Loads q8_1 u and computes sum_u_neg32 = -32*sum(u) once per column across all rows_per_cuda_block,
// eliminating both the second dp4a (at n=1) and the presub bit-twiddles (at n=2..3) per row.
template <int ncols_dst, int rows_per_cuda_block, int vdr>
static __device__ __forceinline__ void p104_mmvq_kloop_q6(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int tid, const int blocks_per_row_x, const int blocks_per_iter,
        const int kbx_offset, const int stride_row_x, const uint32_t stride_col_y,
        float (&tmp)[ncols_dst][rows_per_cuda_block]) {
    constexpr int qk = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qk;
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qi;
    const int kqs = vdr * (tid % (qi / vdr));
    const int bq8_offset   = 2 * QR6_K * (kqs / (QI6_K / 2)) + (kqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (kqs / (QI6_K / 2)) + (kqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift     = 2 * ((kqs % (QI6_K / 2)) / (QI6_K / 4));
    const int qh_offset    = 4 * ((QI6_K / 4) * (kqs / (QI6_K / 2)) + kqs % (QI6_K / 4));
    const int q8_sub       = kqs % QI8_1;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);

        int   u[ncols_dst][QR6_K][vdr];
        int   sum_u_neg32[ncols_dst][QR6_K];
        float d8[ncols_dst][QR6_K];
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                const block_q8_1 * bq8i = &y[c * stride_col_y + kby + bq8_offset + 2 * i];
                const int * q8 = (const int *) bq8i->qs + q8_sub;
                int sneg = P104_MMVQ_MAGIC_I;
#pragma unroll
                for (int j = 0; j < vdr; ++j) {
                    const int uv = __ldg(q8 + j);
                    u[c][i][j] = uv;
                    sneg = ggml_cuda_dp4a(0xE0E0E0E0, uv, sneg);
                }
                sum_u_neg32[c][i] = sneg;
                d8[c][i] = __low2float(__ldg(&bq8i->ds));
            }
        }

#pragma unroll
        for (int r = 0; r < rows_per_cuda_block; ++r) {
            const block_q6_K * bq6_K = (const block_q6_K *) vx + (kbx_offset + r * stride_row_x + kbx);
            int vl[vdr];
            int vh[vdr];
            p104_load_ints_b2<vdr>(bq6_K->ql + 4 * kqs, vl);
            p104_load_ints_b2<vdr>(bq6_K->qh + qh_offset, vh);
            const float d6 = __half2float(__ldg(&bq6_K->d));
            const int8_t * scales = bq6_K->scales + scale_offset;
            float wf[QR6_K];
#pragma unroll
            for (int i = 0; i < QR6_K; ++i) {
                const int sc = __ldg(scales + 4 * i);
                wf[i] = d6 * (__int_as_float(P104_MMVQ_MAGIC_I + sc) - P104_MMVQ_MAGIC_F);
            }

#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                float row_sum = 0.0f;
#pragma unroll
                for (int i = 0; i < QR6_K; ++i) {
                    int sumi = sum_u_neg32[c][i];
#pragma unroll
                    for (int j = 0; j < vdr; ++j) {
                        const int vil = (vl[j] >> (4 * i)) & 0x0F0F0F0F;
                        const int vih = (((vh[j] >> vh_shift) >> (4 * i)) << 4) & 0x30303030;
                        sumi = ggml_cuda_dp4a(vil | vih, u[c][i][j], sumi);
                    }
                    const float fsum = __int_as_float(sumi) - P104_MMVQ_MAGIC_F;
                    row_sum = fmaf(d8[c][i] * wf[i], fsum, row_sum);
                }
                tmp[c][r] += row_sum;
            }
        }
    }
}

// p104 k4: fused k-loop for Q4_K/Q5_K on Pascal (ncols_dst <= 3, no fusion).
// Computes exact q8_1 block sum sumi_m once per column (shared across rows_per_cuda_block),
// processes one weight row at a time to halve register pressure (Q4_K/Q5_K n=3),
// and replaces quarter-rate I2F + XMAD with magic-number float conversion + hoisted float scales.
template <ggml_type type, int ncols_dst, int rows_per_cuda_block>
static __device__ __forceinline__ void p104_mmvq_kloop_q45(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y,
        const int tid, const int blocks_per_row_x, const int blocks_per_iter,
        const int kbx_offset, const int stride_row_x, const uint32_t stride_col_y,
        float (&tmp)[ncols_dst][rows_per_cuda_block]) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = 8;
    const int kqs = vdr * (tid % (qi / vdr));
    const int j_sub = kqs / 8;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);

        float d8[ncols_dst][2];
        float fsum_m[ncols_dst][2];
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
#pragma unroll
            for (int i_sub = 0; i_sub < 2; ++i_sub) {
                const block_q8_1 * bq8i = &y[c * stride_col_y + kby + 2 * j_sub + i_sub];
                const int * q8 = (const int *) bq8i->qs;
                int sumi_m = P104_MMVQ_MAGIC_I;
#pragma unroll
                for (int l = 0; l < QI8_1; ++l) {
                    sumi_m = ggml_cuda_dp4a(0x01010101, __ldg(q8 + l), sumi_m);
                }
                const float d8v = __low2float(__ldg(&bq8i->ds));
                d8[c][i_sub] = d8v;
                fsum_m[c][i_sub] = d8v * (__int_as_float(sumi_m) - P104_MMVQ_MAGIC_F);
            }
        }

#pragma unroll
        for (int r = 0; r < rows_per_cuda_block; ++r) {
            const int row_kbx = kbx_offset + r * stride_row_x + kbx;
            int v[8];
            int vh[8];
            const uint16_t * scales;
            float2 dmf;
            if constexpr (type == GGML_TYPE_Q4_K) {
                const block_q4_K * bq = (const block_q4_K *) vx + row_kbx;
                const int4 * q4 = (const int4 *) (bq->qs + 32 * j_sub);
                const int4 v0 = __ldg(q4 + 0);
                const int4 v1 = __ldg(q4 + 1);
                v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
                v[4] = v1.x; v[5] = v1.y; v[6] = v1.z; v[7] = v1.w;
                scales = (const uint16_t *) bq->scales;
                dmf = __half22float2(__ldg(&bq->dm));
            } else {
                const block_q5_K * bq = (const block_q5_K *) vx + row_kbx;
                const int4 * q5 = (const int4 *) (bq->qs + 32 * j_sub);
                const int4 * qh = (const int4 *) bq->qh;
                const int4 v0 = __ldg(q5 + 0);
                const int4 v1 = __ldg(q5 + 1);
                const int4 h0 = __ldg(qh + 0);
                const int4 h1 = __ldg(qh + 1);
                v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
                v[4] = v1.x; v[5] = v1.y; v[6] = v1.z; v[7] = v1.w;
                vh[0] = h0.x >> (2 * j_sub); vh[1] = h0.y >> (2 * j_sub);
                vh[2] = h0.z >> (2 * j_sub); vh[3] = h0.w >> (2 * j_sub);
                vh[4] = h1.x >> (2 * j_sub); vh[5] = h1.y >> (2 * j_sub);
                vh[6] = h1.z >> (2 * j_sub); vh[7] = h1.w >> (2 * j_sub);
                scales = (const uint16_t *) bq->scales;
                dmf = __half22float2(__ldg(&bq->dm));
            }
            const int jm = j_sub & 1;
            const uint32_t s0 = __ldg(scales + jm + 0);
            const uint32_t s2 = __ldg(scales + jm + 2);
            const uint32_t s4 = __ldg(scales + jm + 4);
            const uint32_t hi = (uint32_t) -(int32_t) (j_sub >= 2);
            const uint32_t a0 = ((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi);
            const uint32_t a1 = ((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi);
            const float wd[2] = {
                dmf.x * (__int_as_float(P104_MMVQ_MAGIC_I | (a0 & 0xff)) - P104_MMVQ_MAGIC_F),
                dmf.x * (__int_as_float(P104_MMVQ_MAGIC_I | ((a0 >> 8) & 0xff)) - P104_MMVQ_MAGIC_F)
            };
            const float wm[2] = {
                dmf.y * (__int_as_float(P104_MMVQ_MAGIC_I | (a1 & 0xff)) - P104_MMVQ_MAGIC_F),
                dmf.y * (__int_as_float(P104_MMVQ_MAGIC_I | ((a1 >> 8) & 0xff)) - P104_MMVQ_MAGIC_F)
            };

#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                float row_sum = 0.0f;
#pragma unroll
                for (int i_sub = 0; i_sub < 2; ++i_sub) {
                    const int * q8 = (const int *) y[c * stride_col_y + kby + 2 * j_sub + i_sub].qs;
                    int sumi_d = P104_MMVQ_MAGIC_I;
#pragma unroll
                    for (int l = 0; l < QI8_1; ++l) {
                        const int u = __ldg(q8 + l);
                        int vl = (v[l] >> (4 * i_sub)) & 0x0F0F0F0F;
                        if constexpr (type == GGML_TYPE_Q5_K) {
                            vl |= (vh[l] << (4 - i_sub)) & 0x10101010;
                        }
                        sumi_d = ggml_cuda_dp4a(vl, u, sumi_d);
                    }
                    const float fd = __int_as_float(sumi_d) - P104_MMVQ_MAGIC_F;
                    row_sum = fmaf(fd, wd[i_sub] * d8[c][i_sub], row_sum);
                    row_sum = fmaf(-wm[i_sub], fsum_m[c][i_sub], row_sum);
                }
                tmp[c][r] += row_sum;
            }
        }
    }
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride, const uint32_t nrows_x) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int vdr = get_vdr_mmvq_tbl(type, table_id, ncols_dst);
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda_tbl(type, table_id, ncols_dst);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    // p104 k1: on Pascal with several rows per block the last block is shifted back so that it only reads rows of the
    // matrix (no read past the end of the tensor). Rows it shares with the previous block are recomputed bit-identically.
    const     int row0 = table_id == MMVQ_PARAMETERS_PASCAL && rows_per_cuda_block > 1 ?
        max(0, min((int) (rows_per_cuda_block*blockIdx.x), (int) nrows_x - rows_per_cuda_block)) : rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    if constexpr (table_id == MMVQ_PARAMETERS_PASCAL && !has_fusion && ncols_dst == 1 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
        p104_mmvq_kloop_q45_n1<type, rows_per_cuda_block>(
            vx, y, tid, blocks_per_row_x, blocks_per_iter, kbx_offset, stride_row_x, tmp);
    } else if constexpr (table_id == MMVQ_PARAMETERS_PASCAL && !has_fusion && type == GGML_TYPE_Q4_K && ncols_dst == 3) {
        p104_mmvq_kloop_q4_col<ncols_dst, rows_per_cuda_block>(
            vx, y, tid, blocks_per_row_x, blocks_per_iter, kbx_offset, stride_row_x, stride_col_y, tmp);
    } else if constexpr (table_id == MMVQ_PARAMETERS_PASCAL && !has_fusion && type == GGML_TYPE_Q5_K && ncols_dst <= P104_WIDE_MAX_NCOLS) {
        p104_mmvq_kloop_q45<type, ncols_dst, rows_per_cuda_block>(
            vx, y, tid, blocks_per_row_x, blocks_per_iter, kbx_offset, stride_row_x, stride_col_y, tmp);
    } else if constexpr (table_id == MMVQ_PARAMETERS_PASCAL && !has_fusion && type == GGML_TYPE_Q6_K && ncols_dst == 1) {
        p104_mmvq_kloop_q6<ncols_dst, rows_per_cuda_block, vdr>(
            vx, y, tid, blocks_per_row_x, blocks_per_iter, kbx_offset, stride_row_x, stride_col_y, tmp);
    } else {
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    ggml_cuda_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, const uint32_t nrows_x, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows_x);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows_x);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

// p104 k1: dispatch trace, compiled in only with GGML_P104_MMVQ_TRACE defined, then enabled by GGML_P104_MMVQ_LOG
static bool p104_mmvq_log() {
#ifdef GGML_P104_MMVQ_TRACE
    static const bool enabled = getenv("GGML_P104_MMVQ_LOG") != nullptr;
    return enabled;
#else
    return false;
#endif
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    // Upstream vdr and nwarps for the small-K decision, so the Pascal wide path switches to small_k at the same K as upstream.
    const int     vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id == MMVQ_PARAMETERS_PASCAL ? MMVQ_PARAMETERS_GENERIC : table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    // p104 k1: the wide Pascal vec_dots use 16-byte loads. Device buffers are >= 128-byte aligned and q4_K/q5_K
    // blocks are 144/176 bytes, so this holds for every block-granular view; a misaligned host buffer aborts here
    // instead of faulting in the kernel (an in-kernel fallback cost up to 50% at 2..3 columns, see LEDGER k2).
    if (table_id == MMVQ_PARAMETERS_PASCAL && mmvq_pascal_wide(type) && ncols_dst <= P104_WIDE_MAX_NCOLS) {
        GGML_ASSERT((((uintptr_t) vx | (uintptr_t) fusion.gate | (uintptr_t) fusion.shared_up | (uintptr_t) fusion.shared_gate) & 15) == 0 &&
                    "p104 wide mmvq path needs 16-byte aligned quantized weights");
    }

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                if (p104_mmvq_log()) {
                    fprintf(stderr, "p104mmvq type=%s k=%d m=%d ncols_dst=1 ids=%d gate=%d table=%d vdr=%d nwarps=%u grid=%u small_k=%d\n",
                        ggml_type_name(type), ncols_x, nrows_x, ids != nullptr, fusion.gate != nullptr, (int) table_id,
                        get_vdr_mmvq_tbl(type, table_id, 1), dims.second.y, dims.first.x, (int) c_small_k);
                }
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    nrows_x, stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, nrows_x, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, nrows_x, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, nrows_x, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, nrows_x, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, nrows_x, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, nrows_x, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, nrows_x, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
