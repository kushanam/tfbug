# tensor placement bug
For the dataset ops, the IteratorBase which is responsible for actually producing data based on other inputs iterators cannot mark its inputs as Host Memory - if the iterator is placed on GPU, all of its inputs are expected to be on the GPU as well, and if it's not it is/it must be transferred there with copy_to_device. By other means the iterator within the dataset op does not respect the placement of tensor of the dataset itself. 

Dataset ops are missing a mechanism to mark inputs to IteratorBase as `HostMemory`. This is needed for IteratorBase to be able to efficiently wrap TensorFlow GPU Ops that have registered HostMemory inputs. Currently, the DataSet API will copy ALL inputs to GPU iterators to the GPU. Then before calling a GPU op with HostMemory inputs, those inputs need to be copied back to the host to vaoid runtime errors.

Let input be on CPU and Dataset op defined as a GPU op. The input is marked as HostMemory upon registration. The iterator within the dataset op does not respect that placement and copies over the data to the GPU.

To run clone the repo then:

```
./build_poc_dataset.sh 
python -u tf_test.py 
```
you should get segmentation fault which is an indication of the desscribed issue.

The line resulting in the segmentation fault is in the `tf_test.py:90`:

```test_copy_to_gpu_gen(process_on='cpu', target_device='/cpu:0')```

This basically runs a pipeline within which the `from_generator` gpu op is utilized to showcase a device dataset op, however the input data is expected to be on the cpu since it is registered as the host memory (`poc_dataset_op.cc:231`). Next within the `poc_dataset_op.cc:175` we try to modify the input as CPU memory. Here is where we get the segfault as the memory has already been copied to the device.
If we modify a the GPU memory instead, everything looks correct (`tf_test.py:91`)





