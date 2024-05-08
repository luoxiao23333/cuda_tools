#include "attn_tools.h"

__global__ void mean_along_dim0_above_threshold_kernel(const float* input, bool* output, int dim0, int dim1, int dim2, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idx * dim2 + idy;
    int step = dim1*dim2;
    if (idx < dim1 && idy < dim2) {
        float sum = 0;
        for (int i = 0; i < dim0; ++i) {
            // sum += input[i * dim1 * dim2 + idx * dim2 + idy];
            sum += input[index];
            index += step;
        }
        sum /= dim0;
        if (sum > threshold) {
            output[idx * dim2 + idy] = 1;
        } else {
            output[idx * dim2 + idy] = 0;
        }
    }
}


at::Tensor mean_above_threshold(const at::Tensor& input, float threshold) {
    auto dim0 = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto above_threshold_output = torch::empty({dim1, dim2}, options.dtype(torch::kBool));

    dim3 blockSize(16, 16);
    dim3 gridSize((dim1 + blockSize.x - 1) / blockSize.x, (dim2 + blockSize.y - 1) / blockSize.y);

    mean_along_dim0_above_threshold_kernel<<<gridSize, blockSize>>>(input.data_ptr<float>(), above_threshold_output.data_ptr<bool>(), dim0, dim1, dim2, threshold);

    return above_threshold_output;
}