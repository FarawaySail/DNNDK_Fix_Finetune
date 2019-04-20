#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib

decent test -model /home/xuyf/test_decent/fix_results/fix_train_test.prototxt -weights /home/xuyf/test_decent/fix_results/fix_train_test.caffemodel -gpu 0 -test_iter 1 1> fix_test.log 2>&1
