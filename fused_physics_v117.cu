// ============================================================
// fused_physics_v117.cu — MaGi v117 Physics Kernel
//
// Computes vel_hb and vel_s for all N workers uniformly.
// No masking — special worker handling done in Python post-call.
// ============================================================

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void magi_physics_v117_kernel(
    const float4* __restrict__ phases_hb,
    const float4* __restrict__ phases_s,
    const float* __restrict__  s_filtered,
    const float* __restrict__  s_deriv,
    const float* __restrict__  s_integral,
    const float* __restrict__  hb_norm,
    const float* __restrict__  hb_deriv_norm,
    const float* __restrict__  hb_int_norm,
    const float4* __restrict__ lens_weights,
    const float* __restrict__  inputs_tensor,
    float4* __restrict__ vel_hb_out,
    float4* __restrict__ vel_s_out,
    const int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ── HB Velocity ───────────────────────────────────────────────────────────
    float hb_mod = 0.15f * hb_deriv_norm[idx]
                 + 0.05f * hb_norm[idx]
                 + 0.03f * hb_int_norm[idx];

    vel_hb_out[idx] = make_float4(
        0.040f + hb_mod,
        0.025f + hb_mod,
        0.015f + hb_mod,
        0.005f + hb_mod
    );

    // ── S-phase Velocity ──────────────────────────────────────────────────────
    float val        = s_filtered[idx];
    float deriv      = s_deriv[idx];
    float integ      = s_integral[idx];
    float grav_scale = 0.02f * (inputs_tensor[idx] / 500.0f);

    float4 lw  = lens_weights[idx];
    float4 phb = phases_hb[idx];
    float4 ps  = phases_s[idx];

    // Dim 0 — Child: Gaussian novelty
    float child_raw = (fabsf(deriv) / 40.0f * lw.x)
                    * __expf(-(deriv/40.0f)*(deriv/40.0f) / 2.0f);
    float vs0 = 0.05f + 0.08f * child_raw + __sinf(phb.x - ps.x) * grav_scale;

    // Dim 1 — Youth: clamped linear ramp
    float youth_raw = fminf(fmaxf(lw.y * (val / 500.0f), 0.0f), 1.0f);
    float vs1 = 0.03f + 0.08f * youth_raw + __sinf(phb.y - ps.y) * grav_scale;

    // Dim 2 — Adult: sigmoid trend prediction
    float youth_c   = fminf(fmaxf(lw.y * (val / 500.0f), 0.0f), 1.0f);
    float adult_inp = (0.6f * youth_c + 0.4f * fabsf(deriv) / 25.0f) - lw.z;
    float adult_raw = fminf(fmaxf(adult_inp / (1.0f + __expf(-8.0f * adult_inp)), 0.0f), 1.0f);
    float vs2 = 0.02f + 0.08f * adult_raw + __sinf(phb.z - ps.z) * grav_scale;

    // Dim 3 — Elder: tanh memory integration
    float elder_raw = ((tanhf((integ - 250.0f) * (4.0f / 300.0f) - 2.0f) + 1.0f) / 2.0f)
                    * lw.w;
    float vs3 = 0.01f + 0.08f * elder_raw + __sinf(phb.w - ps.w) * grav_scale;

    vel_s_out[idx] = make_float4(vs0, vs1, vs2, vs3);
}

void launch_magi_physics_v117(
    const float4* phases_hb, const float4* phases_s,
    const float* s_filtered, const float* s_deriv, const float* s_integral,
    const float* hb_norm, const float* hb_deriv_norm, const float* hb_int_norm,
    const float4* lens_weights, const float* inputs_tensor,
    float4* vel_hb_out, float4* vel_s_out, int N
) {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    magi_physics_v117_kernel<<<blocks, threads>>>(
        phases_hb, phases_s,
        s_filtered, s_deriv, s_integral,
        hb_norm, hb_deriv_norm, hb_int_norm,
        lens_weights, inputs_tensor,
        vel_hb_out, vel_s_out, N
    );
}
