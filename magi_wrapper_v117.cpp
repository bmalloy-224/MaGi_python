#include <torch/extension.h>
#include <cuda_runtime.h>

void launch_magi_physics_v117(
    const float4* phases_hb, const float4* phases_s,
    const float* s_filtered, const float* s_deriv, const float* s_integral,
    const float* hb_norm, const float* hb_deriv_norm, const float* hb_int_norm,
    const float4* lens_weights, const float* inputs_tensor,
    float4* vel_hb_out, float4* vel_s_out, int N
);

std::vector<torch::Tensor> forward(
    torch::Tensor phases_hb,
    torch::Tensor phases_s,
    torch::Tensor s_filtered,
    torch::Tensor s_deriv,
    torch::Tensor s_integral,
    torch::Tensor hb_norm,
    torch::Tensor hb_deriv_norm,
    torch::Tensor hb_int_norm,
    torch::Tensor lens_weights,
    torch::Tensor inputs_tensor
) {
    auto N       = s_filtered.size(0);
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(phases_hb.device());

    auto vel_hb = torch::empty({N, 4}, options);
    auto vel_s  = torch::empty({N, 4}, options);

    launch_magi_physics_v117(
        (const float4*)phases_hb.data_ptr<float>(),
        (const float4*)phases_s.data_ptr<float>(),
        s_filtered.data_ptr<float>(),
        s_deriv.data_ptr<float>(),
        s_integral.data_ptr<float>(),
        hb_norm.data_ptr<float>(),
        hb_deriv_norm.data_ptr<float>(),
        hb_int_norm.data_ptr<float>(),
        (const float4*)lens_weights.data_ptr<float>(),
        inputs_tensor.data_ptr<float>(),
        (float4*)vel_hb.data_ptr<float>(),
        (float4*)vel_s.data_ptr<float>(),
        (int)N
    );

    return {vel_hb, vel_s};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MaGi v117 Physics Fusion");
}
