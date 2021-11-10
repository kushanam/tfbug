#!/bin/bash -ex

 COMPILER=${CXX:-g++}
#COMPILER=${CXX:-clang++}
PYTHON=${PYTHON:-python3}
LIB_NAME=${1:-"libpoc_tf_dataset.so"}
SRCS="poc_dataset_op.cc"
INCL_DIRS="-I/usr/local/cuda/include/"
TF_CFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
$COMPILER -std=c++11 -DNDEBUG -O0 -g3 -shared -fPIC ${SRCS} -o ${LIB_NAME} ${INCL_DIRS} ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
