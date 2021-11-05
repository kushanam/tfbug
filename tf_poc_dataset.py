# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

_poc_tf_module = None

def load_poc_tf_plugin():
    global _poc_tf_module
    if _poc_tf_module is not None:
        return _poc_tf_module

    # asssumming it's compatible
    _poc_tf_module = tf.load_op_library("./libpoc_tf_dataset.so")
    return _poc_tf_module

from tensorflow.python.data.util import nest

load_poc_tf_plugin()


from tensorflow.python.framework import ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
import functools


##
## Dataset Python wrapper
##

def dataset_options():
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = False

    return options

class _PocDatasetV2(dataset_ops.DatasetV2):
    def __init__(self, input_dataset, input_device, output_device, output_dtypes, output_shapes, process_on='cpu', fail_on_device_mismatch=True):

        # Skipping all error checking for tersness

        output_classes = nest.map_structure(lambda _: ops.Tensor, output_dtypes)
        self._input_dataset = input_dataset
        self._input_device = input_device
        self._output_device = output_device
        self._process_on = process_on
        self._output_shapes = output_shapes
        self._output_dtypes = output_dtypes
        self._fail_on_device_mismatch = fail_on_device_mismatch

        self._structure = structure.convert_legacy_structure(self._output_dtypes, self._output_shapes, output_classes)

        super(_PocDatasetV2, self).__init__(self._as_variant_tensor())

    @property
    def element_spec(self):
        return self._structure

    @property
    def _element_structure(self):
        return self._structure

    def _inputs(self):
        # Apparently here TF is happy with a list
        return nest.flatten(self._input_dataset)

    def _as_variant_tensor(self):
        # Call to the Op
        return _poc_tf_module.poc_dataset(
            self._input_dataset._variant_tensor,
            input_device=self._input_device, # TODO(klecki): take this from input dataset placements?
            output_device=self._output_device,
            process_on=self._process_on,
            output_shapes=self._output_shapes,
            output_dtypes=self._output_dtypes,
            fail_on_device_mismatch=self._fail_on_device_mismatch)

_PocDatasetImpl = _PocDatasetV2

class PocDataset(dataset_ops._OptionsDataset):
    @functools.wraps(_PocDatasetV2.__init__)
    def __init__(self, *args, **kwargs):
        dataset_impl = _PocDatasetImpl(*args, **kwargs)
        super(PocDataset, self).__init__(dataset_impl, dataset_options())


