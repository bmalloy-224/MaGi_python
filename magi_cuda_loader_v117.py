"""
magi_cuda_loader_v117.py — Python interface to MaGi v117 physics kernel

Kernel computes vel_hb and vel_s for ALL workers uniformly.
Special worker masking is handled in Python after the kernel returns:
    vel_hb[1548] = 0  — voice: HB driven by voice_carrier.speak()
    vel_s[1548]  = 0  — voice: S-phase driven by voice_carrier.speak()
    vel_s[1549]  = 0  — main BH: S-phase driven by BH logic (HB free to roam)
    vel_s[1551]  = 0  — N BH: S-phase driven by N BH logic (HB free to roam)
"""

import torch


class MagiCUDAv117:
    """Pre-compiled CUDA kernel for MaGi v117 physics fusion."""

    def __init__(self, verbose=True):
        self.module = None

        if verbose:
            print("⚡ Loading MaGi v117 CUDA physics kernel...")

        try:
            import magi_cuda_v117
            self.module = magi_cuda_v117
            if verbose:
                print("⚡ Kernel ready — vel_hb + vel_s, masking handled in Python")
        except ImportError as e:
            if verbose:
                print(f"❌ Failed to load kernel: {e}")
                print("   Run: python compile_v117.py")
                print("   Falling back to Python path...")

    def is_available(self):
        return self.module is not None

    def compute(
        self,
        phases_hb,       # [N, 4] float32 contiguous CUDA
        phases_s,        # [N, 4] float32 contiguous CUDA
        s_filtered,      # [N]    float32 contiguous CUDA
        s_deriv,         # [N]    float32 contiguous CUDA
        s_integral,      # [N]    float32 contiguous CUDA
        hb_norm,         # [N]    float32 contiguous CUDA
        hb_deriv_norm,   # [N]    float32 contiguous CUDA
        hb_int_norm,     # [N]    float32 contiguous CUDA
        lens_weights,    # [N, 4] float32 contiguous CUDA
        inputs_tensor,   # [N]    float32 contiguous CUDA
    ):
        """
        Execute fused physics kernel.
        Returns: (vel_hb [N,4], vel_s [N,4]) — unmasked, all workers.
        Caller applies special worker zeroing in Python after return.
        """
        if self.module is None:
            raise RuntimeError("Kernel not compiled. Run: python compile_v117.py")

        return self.module.forward(
            phases_hb.contiguous(),
            phases_s.contiguous(),
            s_filtered.contiguous(),
            s_deriv.contiguous(),
            s_integral.contiguous(),
            hb_norm.contiguous(),
            hb_deriv_norm.contiguous(),
            hb_int_norm.contiguous(),
            lens_weights.contiguous(),
            inputs_tensor.contiguous(),
        )
