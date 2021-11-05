// Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <chrono>
#include <queue>
#include <sstream>
#include <vector>

#include "tensorflow/core/public/version.h"

#if TF_MAJOR_VERSION == 2 || (TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/framework/tensor.h"

#include "poc_dataset_op.h"

#include "cuda.h"


using namespace tensorflow;        // NOLINT(build/namespaces)
using namespace tensorflow::data;  // NOLINT(build/namespaces)

namespace example_tf_impl {


// internal Dataset, responsible for keeping alive some parameters and implementing the
// MakeIteratorInternal to create another internal class: Iterator : public DatasetIterator<Dataset>
class PocDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext *context, DatasetBase *input,
                   const std::string &input_device,
                   const std::string &output_device,
                   const std::string &process_on,
                   const std::vector<PartialTensorShape> &shapes, const DataTypeVector &dtypes,
                   const bool is_gpu_device, const bool fail_on_device_mismatch)
      : DatasetBase(DatasetContext(context)),
        input_(input),
        input_device_(input_device),
        output_device_(output_device),
        process_on_(process_on),
        shapes_(shapes),
        dtypes_(dtypes),
        device_type_(is_gpu_device ? device_type_t::GPU : device_type_t::CPU),
        fail_on_device_mismatch_(fail_on_device_mismatch) {
    input_->Ref();
    if (is_gpu_device) {
      stream_ = context->eigen_gpu_device().stream();
    }

    LOG(WARNING) << "[INPUT]: context->device->name " << context->device()->name();
  }

  ~Dataset() override {
    input_->Unref();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(const string &prefix) const override;

  const DataTypeVector &output_dtypes() const override {
    return dtypes_;
  }

  const std::vector<PartialTensorShape> &output_shapes() const override {
    return shapes_;
  }

  string DebugString() const override {
    return "example::DatasetOp()::Dataset";
  }

  tensorflow::int64 Cardinality() const override {
    return data::kInfiniteCardinality;
  }

 protected:
  std::vector<PartialTensorShape> shapes_;
  const DataTypeVector dtypes_;
  cudaStream_t stream_ = 0;
  const device_type_t device_type_;
  const bool fail_on_device_mismatch_;

  Status AsGraphDefInternal(SerializationContext *context, DatasetGraphDefBuilder *b,
                            Node **output) const override {
    return tensorflow::errors::Unimplemented("");
  }


#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 3
  Status CheckExternalState() const override {
    return errors::Unimplemented("CheckExternalState is not supported for this dataset.");
  }
#endif

 private:

  class Iterator;

  const DatasetBase *input_;
  const std::string input_device_;
  const std::string output_device_;
  const std::string process_on_;
};


// This is the "Iterator" created by MakeIterator. Via the MakeIterator docstring:
    // This method may be called multiple times on the same instance,
    // and the resulting iterators will have distinct state. Each
    // iterator will traverse all elements in this dataset from the
    // start.
//!!! Note that This class recursively calls the MakeIterator and GetNext on its inputs
class PocDatasetOp::Dataset::Iterator : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const Params &params,
                    bool enable_memory_stats = false)
      : DatasetIterator<Dataset>(params) {}


  // Note here the call to MakeIterator on input dataset
  // The context ~might~ not match the original placement
  Status Initialize(IteratorContext *context) override {
    LOG(WARNING) << "[PocDatasetOp::Dataset::Iterator]: Initialize(IteratorContext *context)" << std::endl;
    mutex_lock l(mu_);

    TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
        context, this, strings::StrCat(prefix(), "[", 0, "]"), &input_impl_));
    return Status::OK();
  }

  // This implements the iteration.
  Status GetNextInternal(IteratorContext *context, std::vector<Tensor> *out_tensors,
                         bool *end_of_sequence) override {
    tensorflow::mutex_lock l(mu_);

    *end_of_sequence = false;


    TF_RETURN_IF_ERROR(input_impl_->GetNext(context, out_tensors, end_of_sequence));


    if (*end_of_sequence) {
      input_impl_.reset();
      return Status::OK();
    }
    auto &tensor = (*out_tensors)[0];
    if (dataset()->process_on_ == "cpu") {
      tensor.flat<int32>().data()[0] = 666;
    } else {
      int tmp = 777;
      cudaMemcpy(tensor.flat<int32>().data(), &tmp, sizeof(int), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      if (cudaGetLastError() != cudaSuccess) {
        return errors::Aborted("CUDA ERROR IN POC OP");
      }
    }

    if (dataset()->input_device_ != dataset()->output_device_) {
      auto &input = (*out_tensors)[0];
      auto output = Tensor(context->allocator({}), DT_INT32, input.shape());
      cudaMemcpy(output.flat<int32_t>().data(), input.flat<int32_t>().data(),
                 input.NumElements() * sizeof(int), cudaMemcpyHostToDevice);
      (*out_tensors)[0] = output;
    }

    return Status::OK();
  }

  ~Iterator() {}

#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 3
  Status SaveInternal(SerializationContext *ctx, IteratorStateWriter *writer) override {
    return errors::Unimplemented("SaveInternal is not supported for this dataset.");
  }

  Status RestoreInternal(IteratorContext *ctx, IteratorStateReader *reader) override {
    return errors::Unimplemented("RestoreInternal is not supported for this dataset");
  }
#endif

 private:

  tensorflow::mutex mu_;
  std::unique_ptr<IteratorBase> input_impl_;
};

void PocDatasetOp::MakeDataset(OpKernelContext *context, DatasetBase **output) {
  DatasetBase* input;
  OP_REQUIRES_OK(context, GetDatasetFromVariantTensor(context->input(0), &input));
  *output = new Dataset(context, input, input_device_, output_device_, process_on_, shapes_,
                        dtypes_, is_gpu_device_, fail_on_device_mismatch_);
}


std::unique_ptr<IteratorBase> PocDatasetOp::Dataset::MakeIteratorInternal(const string &prefix) const {

  return absl::make_unique<Iterator>(Iterator::Params{this, strings::StrCat(prefix, "::PocDataset")});
}


// Regestrations
REGISTER_KERNEL_BUILDER(Name("PocDataset").Device(tensorflow::DEVICE_CPU), PocDatasetOp);

REGISTER_KERNEL_BUILDER(
    Name("PocDataset").Device(DEVICE_GPU).HostMemory("handle").HostMemory("input_dataset"),
    PocDatasetOp);

// TODO(klecki): Is this what we need to do? Based on MapDataset
REGISTER_INPUT_COLOCATION_EXEMPTION("PocDataset");

REGISTER_OP("PocDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("input_device: string")
    .Attr("output_device: string")
    .Attr("process_on: string")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr(
        "output_dtypes: "
        "list({bool, half, float, uint8, uint16, uint32, uint64, int8, int16, int32, int64}) >= 1")
    .Attr("fail_on_device_mismatch: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(Poc Dataset)doc");

}  // namespace example_tf_impl

#endif  // TF_MAJOR_VERSION == 1 && TF_MINOR_VERSION >= 15
