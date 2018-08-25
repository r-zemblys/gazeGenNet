#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:30:12 2017

@author: raimondas
"""

#%% imports
import os, sys, glob, time
import itertools
from distutils.dir_util import mkpath
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rc("axes.spines", top=False, right=False)
plt.ion()

#import seaborn as sns
#sns.set_style("ticks")

###
import argparse
import tensorflow as tf

from model import gazeGenNet as Model
from model import sample_gaussian2d, softmax

from utils_lib.etdata import ETData
from utils_lib.utils import training_params


#%% functions
def get_arguments():
    parser = argparse.ArgumentParser(description='gazeGenNet')
    parser.add_argument('--model_dir', type=str, default='model_final',
                        help='Model directory')
    parser.add_argument('--sample_len', type=int, default=5000,
                       help='The length of one trial')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of trials to generate')
    return parser.parse_args()

#%%setup parameters
args = get_arguments()
args.model_dir = os.path.join('logdir', args.model_dir)
fname_config = os.path.join(args.model_dir, 'config.json')
if os.path.exists(fname_config):
    train_params = training_params(fname_config)
    config = train_params.params
else:
    print "No config file found"
    sys.exit()

config['mode']= 'inference'

fs = float(config["fs"])
max_fix_len = fs*config['max_fix_len']

etdata = ETData()
#%% code
#create model
print("Building model")
model = Model(config)

print("Initialising session")
tf_config = tf.ConfigProto()
tf_config.log_device_placement = False
tf_config.gpu_options.per_process_gpu_memory_fraction=config['per_process_gpu_memory_fraction']
sess = tf.Session(config=tf_config)
init = tf.global_variables_initializer()
sess.run(init)

#model persistence
saver = tf.train.Saver(tf.global_variables(), max_to_keep = 0)
ckpt = tf.train.get_checkpoint_state(args.model_dir)
if ckpt:
    progress_info = os.path.split(ckpt.model_checkpoint_path)[-1].split('-')[1:]
    epoch, global_step = map(int, progress_info)
    print("loading model: %s" % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("No saved model found")
    global_step = 0
    epoch = 0


#%%sample
output_dir = '%s/data.sampleraw/%s-%s'% (args.model_dir,
                                         os.path.split(ckpt.model_checkpoint_path)[-1],
                                         time.strftime("%Y-%m-%d-%H-%M-%S"))
mkpath(output_dir)

print("Starting sampling")
zeros_state = sess.run(model.istate)

for i in tqdm(range(args.n_samples)):
    state = zeros_state

    events = []
    coord = []
    EOS = []
    pis = []

    prev_x = None
    fix_len = 0

    for s in range(args.sample_len):

        #first sample
        if prev_x is None:
            prev_x = np.array([0, 0, 1, 1], dtype=np.float32)

        #feed forward
        feed = {model.input_data: prev_x.reshape(1, 1, -1),
                model.istate: state,}

        fetch = [model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.rho,
                 model.evt,model.eos, model.final_state]
        [pi, mu1, mu2, sigma1, sigma2, rho, evt, eos, state] = sess.run(fetch, feed)

        #store internal outputs
        EOS.append(eos)
        pis.append(pi.squeeze())

        #keep track of fixation duration
        evt = softmax(evt)
        if np.argmax(evt)==0:
            fix_len+=1
        else:
            fix_len = 0

        #sample from the generated gaussians
        idx = np.random.choice(pi.shape[1], p=pi[0])
        x1, x2 = sample_gaussian2d(mu1[0][idx], mu2[0][idx],
                                   sigma1[0][idx], sigma2[0][idx],
                                   rho[0][idx])

        #handle end of sequence (event) flag
        if (eos > 0.5) or (fix_len>max_fix_len):
            eos = 1
        else:
            eos = 0

        #store generated sample
        prev_x = np.array((x1, x2, eos, 1))
        coord.append([x1*config['gaze_scale'], x2*config['gaze_scale']])
        events.append(evt)


    #aggregate generated samples
    events = np.array(events).squeeze()
    trial = np.cumsum(np.array(coord), axis=0)
    _etdata = zip(np.arange(len(trial))/fs,
                  trial[:,0],
                  trial[:,1],
                  itertools.repeat(True),
                  np.roll(np.argmax(events, axis=1)+1, -1)
               )

    etdata.load(np.array(_etdata), **{'source':'np_array'})

    etdata.save('%s/%05d' % (output_dir, i))
    etdata.plot('%s/%05d' % (output_dir, i), save=True, show=False)

    sys.stdout.flush()

