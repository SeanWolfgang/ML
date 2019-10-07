# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:42:10 2019

@author: askin
"""

!git clone https://github.com/aamini/introtodeeplearning_labs.git
%cd introtodeeplearning_labs
!git pull
%cd ..

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time
import functools

import introtodeeplearning_labs as util

is_correct_tf_version = '1.14.0' in tf.__version__
assert is_correct_tf_version, "Wrong tensorflow version ({}) installed.".format(tf.__version__)

is_eager_enabled = tf.executing_eagerly()
assert is_eager_enabled, "Tensorflow eager mode is not enabled"