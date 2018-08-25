# gazeGenNet
A neural network for generating synthetic eye-movement data as described in: 
```sh
@article{zemblys2018gazeNet,
  title={gazeNet: End-to-end eye-movement event detection with deep neural networks},
  author={Zemblys, Raimondas and Niehorster, Diederick C and Holmqvist, Kenneth},
  journal={Behavior research methods},
  year={2018},
}
```
This synthetic eye-tracking data generation network was inspired by the handwritten text generation network [(Graves, 2013)](https://arxiv.org/abs/1308.0850) and used [write-rnn-tensorflow](https://github.com/hardmaru/write-rnn-tensorflow) as the starter code.

## Running gazeGenNet
gazeGenNet was developed on linux using Python 2.7 and `Tensorflow 0.12.1`. Code also requires the following libraries:
```sh
numpy
scipy
matplotlib
pandas
tqdm
```
#### TODO
- prepare a docker file
### Sampling from a pretrained model
To sample from a pretrained model that was used in [Zemblys et al. (2018)]() run:
```sh
python run_sampling.py
```
By default it will generate 100 trials, each 5000 samples (10 seconds) long and plot position over time and scanpath plots. The output is written to `logdir/model_final/data.sampleraw/model-210-2000-YYYY-mm-dd-HH-MM-SS` directory. You can pass the following arguments to the script to generate more/less trials of different duration, or use your own trained model:
```sh
  --model_dir MODEL_DIR
                        Model directory
  --sample_len SAMPLE_LEN
                        The length of one trial
  --n_samples N_SAMPLES
                        Number of trials to generate
```

### Training a new model
To train a new model first prepare your training data. Data needs to be structured numpy arrays with a following format:
```
dtype = np.dtype([
	('t', np.float64),	#time in seconds
	('x', np.float32),	#horizontal gaze direction in degrees
	('y', np.float32), 	#vertical gaze direction in degrees
	('status', np.bool),	#status flag. False means trackloss 
	('evt', np.uint8)	#event label:
					#0: Undefined
					#1: Fixation
					#2: Saccade
					#3: Post-saccadic oscillation
					#4: Smooth pursuit
					#5: Blink
])
```
Then create a model directory in `./logdir/`, for example `model_dev` Make sure to have a `config.json` file here and run:
```sh
python run_training.py --model_dir model_dev
```
You can adjust various training parameters by editing the `config.json` file:
```sh
{
    "data_train": "data.unpaired_clean_augment.pkl",  #a pickle of list of structured numpy arrays
                                                      #can also be a training data folder, containing structured numpy arrays; experimental
    									
    "seq_len": 100,                                   #sequence length; ignored if split_seqs is false
    "split_seqs": true,                               #if true, splits data trials into sequences of seq_len+2
    "events": [1, 2, 3],                              #which events are used for training
    "augment": true,                                  #if true, white noise is added to each of the training sequences
    "augment_noise": [0, 0.005, 0.005],               #additive noise level
    "gaze_scale": 1.0,                                #scales input data
    "fs": 500,                                        #data sampling rate
    "max_fix_len": 0.5,                               #maximum fixation length; used to set end of event flag to 1 when sampling
    
    "num_epochs": 100000,                             #number of epochs to train
    "num_steps": 2000,                                #number of steps to train
    "save_every": 250,                                #number of steps after which model is saved
    "info_every": 10,                                 #number of steps for info output
    "batch_size": 50, 
    "max_to_keep": 1,                                 #how many models to keep
    "per_process_gpu_memory_fraction": 0.1,           #fraction of GPU memory to use
    
    "model": "lstm",                                  #model type; not used
    "num_layers": 3,                                  #number of sequence-to-sequence layers
    "rnn_size": 128,                                  #number of neurons in each layer
    "nmixtures": 20,                                  #number of Gaussians to use
    "grad_clip": 10.0, 
    "keep_prob": 0.75,                                #1-dropout
     
    "learning_rate": 0.001,
    "optimizer": "rmsprop",                           #rmsprop or adam (experimental)
    
    #optimizer parameters
    "decay": 0.9, 
    "momentum": 0.9 
}

``` 
#### TODO
- provide instructions on how to prepare original training data
