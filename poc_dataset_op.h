// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TF_PLUGIN_DATASET_H_
#define TF_PLUGIN_DATASET_H_

#include <chrono>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/core/public/version.h"

#if TF_MAJOR_VERSION == 2 || (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"


namespace example_tf_impl {

typedef enum {
  CPU = 0,
  GPU = 1
} device_type_t;


// The op Kernel, it implements the MakeDataset called in tensorflow::data::DatasetOpKernel::Compute
// that creates instance of internal Dataset, that derives from tensorflow::data::DatasetBase
class PocDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  explicit PocDatasetOp(tensorflow::OpKernelConstruction* context)
      : DatasetOpKernel(context),
        is_gpu_device_(context->device_type() == "GPU"),
        context_(context) {
    OP_REQUIRES_OK(context, context->GetAttr(kInputDevice, &input_device_));
    OP_REQUIRES_OK(context, context->GetAttr(kOutputDevice, &output_device_));
    OP_REQUIRES_OK(context, context->GetAttr(kProcessOn, &process_on_));
    OP_REQUIRES_OK(context, context->GetAttr(kOutputShapes, &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr(kOutputDtypes, &dtypes_));
    OP_REQUIRES_OK(context, context->GetAttr(kDeviceMismatch, &fail_on_device_mismatch_));
  }

  void MakeDataset(tensorflow::OpKernelContext* context,
                   tensorflow::data::DatasetBase** output) override;

 private:

  // Arguments describing inputs
  static constexpr const char* const kInputDevice = "input_device";
  static constexpr const char* const kOutputDevice = "output_device";
  static constexpr const char* const kProcessOn = "process_on";


  // Arguments describing outputs
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kOutputDtypes = "output_dtypes";

  // DatasetOp-specific arguments
  static constexpr const char* const kDeviceMismatch = "fail_on_device_mismatch";


  std::string input_device_;
  std::string output_device_;
  std::string process_on_;
  std::vector<tensorflow::PartialTensorShape> shapes_;
  tensorflow::DataTypeVector dtypes_;
  bool is_gpu_device_;
  bool fail_on_device_mismatch_;
  tensorflow::OpKernelConstruction* context_;

  // This is the internal Dataset class, defined in poc_dataset_op.cc
  class Dataset;
};


}  // namespace example_tf_impl

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15

#endif  // TF_PLUGIN_DATASET_H_
