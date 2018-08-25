import numpy as np
import tensorflow as tf

class gazeGenNet():
    def __init__(self, args):

        #%% model params
        self.rnn_size = args['rnn_size']
        self.train = True if args['mode'] == 'train' else False

        self.nmixtures = args['nmixtures']
        self.batch_size = args['batch_size'] if self.train else 1 # training/sampling specific
        self.tsteps = args['seq_len'] if self.train else 1 # training/sampling specific

        # training params
        self.grad_clip = args['grad_clip']

        # other
        self.evt_vec_len = len(args['events'])
        self.graves_initializer = tf.truncated_normal_initializer(mean=0.,
                                                                  stddev=.075,
                                                                  seed=None,
                                                                  dtype=tf.float32)
        self.window_b_initializer = tf.truncated_normal_initializer(mean=-3.0,
                                                                    stddev=.25,
                                                                    seed=None,
                                                                    dtype=tf.float32)
        input_shape = 4 #(x, y), eos, mode

        self.input_vec_dim = input_shape
        self.output_vec_dim = 3+self.evt_vec_len #(x, y), eos, evt

        #%% build the basic recurrent network architecture
        cell_func = tf.nn.rnn_cell.LSTMCell # could be GRUCell or RNNCell

        cell = cell_func(args['rnn_size'])
        if (self.train and args['keep_prob'] < 1): # training mode
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = args['keep_prob'])

        cell_multi = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * args['num_layers'],
            state_is_tuple=True
        )

        if (self.train and args['keep_prob'] < 1): # training mode
            cell_multi = tf.nn.rnn_cell.DropoutWrapper(cell_multi,
                                                       output_keep_prob = args['keep_prob'])

        #define placeholders for input, output and states
        self.input_data = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.tsteps, self.input_vec_dim])
        self.target_data = tf.placeholder(dtype=tf.float32,
                                          shape=[None, self.tsteps, self.output_vec_dim])

        self.istate = cell_multi.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        #slice the input volume into separate vols for each tstep
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.tsteps, self.input_data)]
        self.inputs = inputs
        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs,
                                                        self.istate,
                                                        cell_multi,
                                                        loop_function=None,
                                                        scope='rnnlm')
        self.outputs = outputs

        #%% Mixture Density Network. Dense layer to predict the MDN params
        # params = evt, eos + 6 parameters per Gaussian
        n_out = self.evt_vec_len + 1 + self.nmixtures * 6
        with tf.variable_scope('mdn_dense'):
            mdn_w = tf.get_variable("output_w",
                                    [self.rnn_size, n_out],
                                    initializer=self.graves_initializer)
            mdn_b = tf.get_variable("output_b",
                                    [n_out],
                                    initializer=self.graves_initializer)

        #concat outputs for efficiency
        output = tf.reshape(tf.concat(1, outputs), [-1, args['rnn_size']])
        output = tf.nn.xw_plus_b(output, mdn_w, mdn_b) #data flows through dense nn
        self.final_state = last_state
        self.output = output

        #build mixture density cap on top of second recurrent cell
        def gaussian2d(x1, x2, mu1, mu2, s1, s2, rho):
            # define gaussian mdn (eq 24, 25 from http://arxiv.org/abs/1308.0850)
            x_mu1 = tf.subtract(x1, mu1)
            x_mu2 = tf.subtract(x2, mu2)
            Z = tf.square(tf.div(x_mu1, s1)) + \
                tf.square(tf.div(x_mu2, s2)) - \
                2*tf.div(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.multiply(s1, s2))
            rho_square_term = 1-tf.square(rho)
            power_e = tf.exp(tf.div(-Z,2*rho_square_term))
            regularize_term = 2*np.pi*tf.multiply(tf.multiply(s1, s2), tf.sqrt(rho_square_term))
            gaussian = tf.div(power_e, regularize_term)
            return gaussian

        def get_loss(pi, x1_data, x2_data, eos_data, evt_data, mu1, mu2, sigma1, sigma2, rho, eos, evt):
            # define loss function (eq 26 of http://arxiv.org/abs/1308.0850)
            gaussian = gaussian2d(x1_data, x2_data, mu1, mu2, sigma1, sigma2, rho)
            term1 = tf.multiply(gaussian, pi)
            term1 = tf.reduce_sum(term1, 1, keep_dims=True) #do inner summation
            term1 = -tf.log(tf.maximum(term1, 1e-20)) # some errors are zero -> numerical errors.

            term2 = tf.multiply(eos, eos_data) + tf.multiply(1-eos, 1-eos_data) #modified Bernoulli -> eos probability
            term2 = -tf.log(tf.maximum(term2, 1e-20)) #negative log error gives loss

            term3 = tf.nn.sigmoid_cross_entropy_with_logits(evt, evt_data, name=None)

            return term1, term2, term3

        #transform dense NN outputs into params for MDN
        def get_mdn_coef(Z):
            # returns the tf slices containing mdn dist params (eq 18...23 of http://arxiv.org/abs/1308.0850)
            eos_hat = Z[:, 0:1] #end of event tokens
            evt_hat = Z[:, 1:self.evt_vec_len+1] #evt

            pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(1, 6, Z[:, self.evt_vec_len+1:])
            self.pi_hat, self.sigma1_hat, self.sigma2_hat = \
                                        pi_hat, sigma1_hat, sigma2_hat # these are useful for bias method during sampling

            eos = tf.sigmoid(1*eos_hat)
            pi = tf.nn.softmax(pi_hat) # softmax z_pi:
            mu1 = mu1_hat; mu2 = mu2_hat # leave mu1, mu2 as they are
            sigma1 = tf.exp(sigma1_hat); sigma2 = tf.exp(sigma2_hat) # exp for sigmas
            rho = tf.tanh(rho_hat) # tanh for rho (squish between -1 and 1)

            return [eos, evt_hat, pi, mu1, mu2, sigma1, sigma2, rho]

        #%% get output
        flat_target_data = tf.reshape(self.target_data,[-1, self.output_vec_dim])
        self.flat_target_data = flat_target_data
        [x1_data, x2_data, eos_data, evt_data] = tf.split_v(flat_target_data, [1, 1, 1, self.evt_vec_len], 1)

        [self.eos, self.evt, self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho] = get_mdn_coef(output)

        self.losses = get_loss(self.pi, x1_data, x2_data, eos_data, evt_data, \
                               self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, \
                               self.eos, self.evt)
        loss = tf.reduce_sum(sum(self.losses))
        self.cost = loss / (self.batch_size * self.tsteps)

        #%%bring together all variables and prepare for training
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.decay = tf.Variable(0.0, trainable=False)
        self.momentum = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)

        if args['optimizer'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif args['optimizer'] == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum)
        else:
            raise ValueError("Optimizer type not recognized")
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

def sample_gaussian2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def softmax(x, axis = None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis).reshape(-1,1))
    return e_x / e_x.sum(axis=axis).reshape(-1,1)