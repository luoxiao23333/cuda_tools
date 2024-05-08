
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "attn_tools.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_above_threshold", &mean_above_threshold, "Mean and Threshold (CUDA)");
}
