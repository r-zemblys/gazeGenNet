#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:30:12 2017

@author: raimondas
"""

#%% imports
from __future__ import print_function

import os, sys, glob, time
import itertools
from distutils.dir_util import mkpath
from tqdm import tqdm
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#import seaborn as sns
#sns.set_style("ticks")

##
import tensorflow as tf
import pandas as pd

import argparse
import pickle

from utils_lib.utils import DataReader, training_params
from model import gazeGenNet as Model

#%% functions


#%% setup parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='model_dev',
                   help='Directory to save the model')
args = parser.parse_args()

fname_config = os.path.join('logdir', args.model_dir, 'config.json')
if os.path.exists(fname_config):
    train_params = training_params(fname_config)
    config = train_params.params
    for key, val in config.iteritems():
        parser.add_argument('--%s'%key, type=type(val), default = val)
    args = parser.parse_args()
    args.model_dir = os.path.join('logdir', args.model_dir)
else:
    print("No config file found in %s" % args.model_dir)
    sys.exit()

config['mode'] = 'train'
#%% create model
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
saver = tf.train.Saver(tf.global_variables(), max_to_keep = config["max_to_keep"])
ckpt = tf.train.get_checkpoint_state(args.model_dir)
#ckpt = False
if ckpt:
    progress_info = os.path.split(ckpt.model_checkpoint_path)[-1].split('-')[1:]
    epoch, global_step = map(int, progress_info)
    print("loading model: %s" % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("No saved model found")
    global_step = 0
    epoch = 0

#%% data reader
if os.path.splitext(config['data_train'])[-1] == '.pkl':
    #data needs to be a pickle of seq_len+2 length etdata arrays
    fname_data_train = '%s/%s' % (args.model_dir, config['data_train'])
    if os.path.exists(fname_data_train):
        with open(fname_data_train, 'rb') as f:
            train_data = pickle.load(f)
    else:
        print("Run data preparation. Exiting")
        sys.exit()
else:
    #!!! EXPERIMENTAL !!!
    FILES = glob.glob('%s/*.npy'%config['data_train'])
    train_data = []
    for _fpath in FILES:
        _data = np.load(_fpath)
        _mask = np.in1d(_data['evt'], config['events'])
        _data['status'][~_mask] = False

        train_data.append(_data)

reader_train = DataReader(train_data,
                          config=config,
                          epoch=epoch,
                          coord=tf.train.Coordinator(),
                          queue_size=config['batch_size']*5)
dequeue_op_train = reader_train.dequeue(config['batch_size'])

reader_train.epoch = epoch
reader_train.start_threads(sess)


#%%training
print("Starting training")
zeros_state = sess.run(model.istate)
sess.run(tf.assign(model.learning_rate, config['learning_rate']))
sess.run(tf.assign(model.decay, config['decay']))
sess.run(tf.assign(model.momentum, config['momentum']))

epoch_val = 0
train_dur = []

while (epoch < config['num_epochs']) and (global_step < config["num_steps"]):

    epoch+=1
    while True:
        global_step+=1

        #run training step
        start = time.time()
        x, y, sl, trid, e = sess.run(dequeue_op_train)
        #e_dequeue = time.time()

        feed = {model.input_data: x, model.target_data: y,
                model.istate: zeros_state}
        [train_loss, losses, _] = sess.run([model.cost, model.losses, model.train_op], feed)
        losses_fmt = map(lambda s: '%.3f'%s, itertools.chain.from_iterable([np.mean(l, axis=0) for l in losses]))
        losses_str = ', '.join(losses_fmt)

        end = time.time()
        train_dur.append(end-start)
        #print (e_dequeue - start)

        #print info
        if (global_step % config['info_every'] == 0) or (not np.isfinite(train_loss)):
            info = (epoch, global_step, train_loss, losses_str, np.mean(train_dur),config['learning_rate'])
            print("Epoch: %d, Step: %d, loss: %.3f, losses: %s, duration: %.3f, lr: %f" % info)
            with open('%s/log.log' % args.model_dir, 'a') as f:
                f.write("Epoch: %d, Step: %d, loss: %.3f, losses: %s, duration: %.3f, lr: %f\n" % info)

            train_dur = []

        if not np.isfinite(train_loss):
            print ("Something went wrong. Exiting...")
            sys.exit()


        if global_step % config['save_every'] == 0:
            print("Saving model...")
            saver.save(sess, os.path.join(args.model_dir, 'model-%d'%epoch), global_step = global_step)
            print("...done")

        sys.stdout.flush()

        #update epoch info
        if (e > epoch).any():
            break
