#pragma once

#include <torch/extension.h>

void launch_forward_substitution_cuda(torch::Tensor L, torch::Tensor A, int BT);
