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

import os
import tensorflow as tf
import numpy as np

from tf_poc_dataset import PocDataset




iters=2


class FixedSampleIterator:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        return self

    def __next__(self):
        return self.value


def gen_dataset():
    dataset = tf.data.Dataset.from_generator(FixedSampleIterator,
        output_types=tf.int32,
        output_shapes=(None, None),
        args=(np.full((2, 3), 42, np.int32),))
    return dataset



# 1. simple pass-through with CPU-placed datasets
def test_cpu_to_cpu_gen(process_on='cpu'):
    with tf.device('/cpu:0'):
        dataset = gen_dataset()
        poc_dataset = PocDataset(dataset, "cpu", "cpu", (tf.int32,), ((None,),), process_on)
    for i, data in zip(range(iters), poc_dataset):
        print(data)

# 2. Everything is placed on GPU, and this is a bit weird - for the `from_tensor` the data still
# arrives on CPU. I don't think there is a TensorsDataset for GPU registered
def test_gpu_to_gpu_gen(process_on='cpu', output_device='gpu'):
    with tf.device('/gpu:0'):
        dataset = gen_dataset()
        # if we don't make copy the output data to GPU tf errors out
        poc_dataset = PocDataset(dataset, "cpu", output_device, (tf.int32,), ((None,),), process_on)
        for i, data in zip(range(iters), poc_dataset):
            print(data)

# 3. Directly pass from CPU to GPU dataset
def test_cpu_to_gpu_direct_gen(process_on='gpu', output_device='gpu'):
    with tf.device('/cpu:0'):
        dataset = gen_dataset()

    with tf.device('/gpu:0'):
        poc_dataset = PocDataset(dataset, "cpu", output_device, (tf.int32,), ((None,),), process_on)
    for i, data in zip(range(iters), poc_dataset):
        print(data)

# 4. and 5. CPU -> GPU with copy_to_device (note that target_device)
def test_copy_to_gpu_gen(process_on='gpu', target_device='/cpu:0'):
    with tf.device('/cpu:0'):
        dataset = gen_dataset()
        dataset = dataset.apply(tf.data.experimental.copy_to_device(target_device))

    with tf.device('/gpu:0'):
        poc_dataset = PocDataset(dataset, "gpu", "gpu", (tf.int32,), ((None,),), process_on)
    for i, data in zip(range(iters), poc_dataset):
        print(data)


def gen_cases():

    # Remote Calls in copy_to_device allow us to call CPU-placed GeneratorDataset from GPU dataset.
    test_copy_to_gpu_gen(process_on='cpu', target_device='/cpu:0') # error, can't touch data on CPU
    test_copy_to_gpu_gen(process_on='gpu', target_device='/cpu:0') # but it works on GPU


if __name__ == "__main__":
    # simple_cases()
    gen_cases()
