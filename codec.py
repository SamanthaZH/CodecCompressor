import Codec.load_ckpt as clc
import os


root = 'log2/fd1_lego/lego_f'
scene = 3

planes = clc.batch_loading(root,scene)
