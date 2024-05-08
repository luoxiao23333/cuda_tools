import torch
from cuda_tools.attn import mean_above_threshold # THIS LIB MUST BE IMPORTTED BEFORE torch

# at::Tensor mean_above_threshold(const at::Tensor& input, float threshold);
# @param 3 dim float32 tensors
# @return 2 dim torch.bool tensors

attn_map = torch.rand((64, 600, 600), dtype=torch.float32, device="cuda")
threshold = 0.5
result = mean_above_threshold(attn_map, threshold)
print(attn_map.mean(dim=0))
print(result)