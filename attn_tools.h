
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

at::Tensor mean_above_threshold(const at::Tensor& input, float threshold);