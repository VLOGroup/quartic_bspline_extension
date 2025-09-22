#pragma once

#include <vector>
#include <torch/extension.h>

inline void check_device_and_datatype(const std::vector<torch::Tensor>& tensors){
    auto device_0 = tensors[0].device();
    auto type_0 = tensors[0].scalar_type();

    for (size_t j = 1; j < tensors.size(); j++){
        TORCH_CHECK(tensors[j].device() == device_0, 
            "Tensors must be on the same device.");
        TORCH_CHECK(tensors[j].scalar_type() == type_0, 
            "Tensors must have the same data type.");
    }
}