#!/bin/bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/samim/lib/libtorch ..
make
