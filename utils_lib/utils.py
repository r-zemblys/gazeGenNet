import random
import threading
import json

import numpy as np
import tensorflow as tf

#from .etdata import ETData

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def round_up_to_odd(f, min_val = 3):
    """Rounds input value up to nearest odd number.
    Parameters:
        f       --  input value
        min_val --  minimum value to retun
    Returns:
        Rounded value
    """
    w = np.int32(np.ceil(f) // 2 * 2 + 1)
    w = min_val if w < min_val else w
    return w


def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def BoxMuller_gaussian(u1,u2):
  z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
  z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
  return z1,z2


def data_iterator(gaze_data, config, epoch = 0):


    seq_len = config['seq_len']+1 #input is in fact diff(input), therefore we want +1 sample
    split_seqs = config['split_seqs']
    augment = config['augment']
    mode = config['mode']
#    etdata = ETData()

    #augmentation noise levels
    rms_noise_levels = np.arange(*config["augment_noise"])

    #infine loop
    while True:
        epoch+=1

        #shuffle data if training
        if mode == 'train':
            random.shuffle(gaze_data)

        #iterate through the data
        for trid, d in enumerate(gaze_data): #iterates over files


            #for the future improvements; only required if data comes in "raw" format
            #does not affect "prepared"  data
            dd = np.split(d, np.where(np.diff(d['status'].astype(np.int0)) != 0)[0]+1)
            dd = [_d for _d in dd if (_d['status'].all() and len(_d) > seq_len)]

            for seq in dd: #iterates over chunks of valid data
                if split_seqs and len(seq) > seq_len+1:
                    seqs = [seq[pos:pos + seq_len + 1] if (pos + seq_len + 1) < len(seq)-1 else
                            seq[len(seq)-seq_len-1:len(seq)] for pos in xrange(0, len(seq)-1, seq_len/2)]
                else:
                    seqs = [seq]

                for s in seqs:
                    #skips trials with only fixations
                    if (s['evt'] == 1).all():
                        continue

                    #get event of previous sample
                    evt = convertToOneHot(s['evt']-1, len(config['events']))

                    #swap x and y
                    inpt_dir = ['x', 'y']
                    if augment:
                        random.shuffle(inpt_dir)
                    gaze_x = np.copy(s[inpt_dir[0]])
                    gaze_y = np.copy(s[inpt_dir[1]])
                    outpt_dir = inpt_dir


                    gaze_outpt_x = np.copy(s[outpt_dir[0]])
                    gaze_outpt_y = np.copy(s[outpt_dir[1]])
                    #add noise
                    if augment:
                        u1, u2 = np.random.uniform(0,1, (2, len(s)))
                        noise_x, noise_y = BoxMuller_gaussian(u1,u2)
                        rms_noise_level = np.random.choice(rms_noise_levels)
                        noise_x*=rms_noise_level/2
                        noise_y*=rms_noise_level/2
                        #rms = np.sqrt(np.mean(np.hypot(np.diff(noise_x), np.diff(noise_y))**2))
                        gaze_x+=noise_x
                        gaze_y+=noise_y

                    #TODO. Does not do anything special; left for future extensions
                    if mode == 'train':
                        #we train generator and predictor at the same time
                        m = np.round(np.random.rand())
                    elif mode in ['inference']:
                        m = 1.

                    inpt_x, inpt_y, eos = [np.diff(gaze_x)/config["gaze_scale"],
                                           np.diff(gaze_y)/config["gaze_scale"],
                                           np.diff(s['evt']).astype(np.bool)]
                    outpt_x, outpt_y    = [np.diff(gaze_outpt_x)/config["gaze_scale"],
                                           np.diff(gaze_outpt_y)/config["gaze_scale"],]

                    X = zip(inpt_x[:-1],
                            inpt_y[:-1],
                            eos[:-1]*m,
                            np.ones(len(s))*m)

                    X = np.array(X, dtype=np.float32)
                    Y = [(_x, _y, _eos) + (_e) for _x, _y, _eos, _e in zip(outpt_x[1:],
                                                                           outpt_y[1:],
                                                                           eos[1:],
                                                                           map(tuple, evt[1:-1]) )]
                    Y = np.array(Y, dtype=np.float32)

                    yield [X, Y, len(X), trid, epoch]


class DataReader(object):
    '''Generic background reader that preprocesses files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data,
                 config,
                 epoch,
                 coord,
                 queue_size = 1,
                 epochs = 10,
        ):
        self.data = data
        self.config = config
        self.epoch = epoch
        self.coord = coord
        self.epochs = epochs
        self.threads = []

        self.x_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.y_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.sl_placeholder = tf.placeholder(dtype=tf.int32, shape=None)

        #TODO: fix trialID
        self.trialId_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.e_placeholder = tf.placeholder(dtype=tf.int32, shape=None)

        input_shape = 4

        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'float32', 'int32', 'int32', 'int32'],
                                         shapes=[(None, input_shape), (None, 3+len(config['events'])), (), (), ()],
                                         )
        self.enqueue = self.queue.enqueue([self.x_placeholder,
                                           self.y_placeholder,
                                           self.sl_placeholder,
                                           self.trialId_placeholder,
                                           self.e_placeholder,
                       ])

    def dequeue(self, num_elements):
        #TODO: check if is alive
        x, y, sl, trialId, e = self.queue.dequeue_many(num_elements)
        return [x, y, sl, trialId, e]

    def thread_main(self, sess):
        iterator = data_iterator(self.data, self.config, self.epoch)
        for n, (x, y, sl, trialId, epoch) in enumerate(iterator):
            self.epoch = epoch
            if self.coord.should_stop():
                break
            sess.run(self.enqueue, feed_dict={self.x_placeholder: x,
                                              self.y_placeholder: y,
                                              self.sl_placeholder: sl,
                                              self.trialId_placeholder: trialId,
                                              self.e_placeholder: epoch,
                                              })

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

class training_params(object):
    def __init__(self, param_file):
        self.param_file = param_file
        self.read_params()

    class bcolors(object):
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    def read_params(self, current_params = None):
        changed = False
        with open(self.param_file, 'r') as f:
            self.params = json.load(f)
        if not(current_params is None) and not (current_params == self.params):
            print "TRAINING PARAMETERS CHANGED"
            changed = True
            for k, p in current_params.iteritems():
                if not(p == self.params[k]):

                    print self.bcolors.WARNING + \
                          "%s: %s --> %s" % (k, p, self.params[k]) + \
                          self.bcolors.ENDC
        return changed