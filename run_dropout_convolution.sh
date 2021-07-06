#!/bin/bash

. /home/marcel/Mindspore/kungfu-mindspore/ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path /home/marcel/Mindspore/kungfu-mindspore/mindspore)

python dropout_convolution.py
