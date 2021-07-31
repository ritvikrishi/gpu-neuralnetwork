# Feedforward neural network on GPU

Implemention of a neural network in GPU using CUDA and comparing the runtimes with that of CPU. 

## INSTRUCTIONS TO RUN THE CODE:

1) Download the MNIST dataset by running the following commands in a linux terminal:
	`$ python dataset.py` or `$ python3 dataset.py`
  (this might take a few minutes)

2) Compile the source files with:	`$ make`

3) Run the code as :
  `$ USAGE: ./a.out -u <use> -h <num_hidden_layers> -n <size_hlayer> -e <epochs> -s <num_samples> -l <learning_rate> -b <batch_size> -o <optimizer> -a <activation_func>`
  where:
  
  	`<use>` - can be 'cpu' or 'gpu' or 'both' and defines what should be used. (default = 'both')
  
  	`<num_hidden_layers>` - is the number of layers. (default = 2)
  	
	`<size_hlayer>` - is the size of each hidden layer. (default = 128)
  	
	`<epochs>` - is the number of epochs to run. (default = 3)
  	
	`<num_samples>` - is the number of training samples to train the model on. (default = 10000)
  	
	`<learning_rate>` - is the learning rate for the optimizer. (default = 0.001)
  	
	`<batch_size>` - is the mini-batch size. (default = 32)
  	
	`<optimizer>` - is the optimizer and can be 'sgd' or 'mbgd' or 'rmsprop' or 'adam'. (default = "mbgd")
  	
	`<activation_func>` - is the activation function for the neuron and can be 'sigmoid' or 'tanh' or 'relu'. (default = "relu")
  
  All these parameters are optional and can be input in any order.
  Running on default parameters takes around 25 minutes.

4) To remove the executables, run 
	`$ make clean`
