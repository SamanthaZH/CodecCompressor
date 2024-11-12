import Codec.load_ckpt as clc
import os


root = 'log/fd1_worker/worker_f'
scene = 2

planes = clc.batch_loading(root,scene)
