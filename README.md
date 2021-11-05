# tensor placement bug
For the dataset ops, the IteratorBase which is responsible for actually producing data based on other inputs iterators cannot mark its inputs as Host Memory - if the iterator is placed on GPU, all of its inputs are expected to be on the GPU as well, and if it's not it is/it must be transferred there with copy_to_device. By other means the iterator within the dataset op does not respect the placement of tensor of the dataset itself. 

Let input be on CPU and Dataset op defined as a GPU op. The input is marked as HostMemory upon registration. The iterator within the dataset op does not respect that placement and copies over the data to the GPU.

To run clone the repo then:

```
./build_poc_dataset.sh 
python -u tf_test.py 
```
you should get segmentation fault which is an indication of the desscribed issue.



