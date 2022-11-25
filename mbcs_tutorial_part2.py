# Methods for Brain and Cognitive Sciences - Mathematical Modeling Part 2

# install neurogym (for training environments)
# ! git clone https://github.com/neurogym/neurogym.git
# %cd neurogym/
# ! pip install -e .

import time                 # for measuring time
import numpy as np          # for numerical computation 
from matplotlib import cm   # for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D

import gym                  # for neurogym (a parent package)
import neurogym as ngym     # for training environments
import torch                # for training neural nets
import torch.nn.functional as F

from sklearn.decomposition import PCA # for principal component analysis
from scipy.special import softmax
mpl.rcParams.update(mpl.rcParamsDefault)


# Parameter definition
hpar = dict(
	## Task parameters
	dt             = 20,        # simulation time step [ms]
	stim_len       = 1000,      # stimulus input duration [ms] 
	stim_noise     = 0.3,       # stimulus input noise [unitless]
	seq_len        = 60,        # trial duration in time steps

	## Network parameters
	n_neurons      = 64,        # number of recurrent neurons
	network_noise  = 0.1,       # noise to network [unitless]
	tau            = 100,       # network time constant [ms]
	activation_fun = 'relu',    # activation function (sigmoid, relu, softplus)

	## Network parameters: training
	batch_size     = 16,        # size of batch (#trials processed before model is one-step trained)
	n_iteration    = 1000,      # number of epochs (#iterations for training model) 
	learning_rate  = 0.01,      # training learning rate
	optimizer      = 'Adam',        # optimizer (Adam, RMSprop, SGD)    
	loss           = 'CrossEntropy' # loss function (CrossEntropy, L1, L2)    
)


# Learning problem: Task definition
## make neurogym environment
task_name = 'PerceptualDecisionMaking-v0' 
kwargs = {'dt': hpar['dt'], 
		  'timing': {'stimulus': hpar['stim_len']}, 
		  'sigma': hpar['stim_noise']}
env = gym.make(task_name, **kwargs) # boilerplate gym

## make supervised dataset
dataset    = ngym.Dataset(env, batch_size=hpar['batch_size'], seq_len=hpar['seq_len'])

## generate one batch of data when called
inputs, target = dataset()
print('Input to network has shape (SeqLen, Batch, Dim) =', inputs.shape)
print('Target to network has shape (SeqLen, Batch) =', target.shape)


# Learning problem: Task structure
## Randomly select one example trial (=input-output pair) from the task environment.
i_trial = np.random.choice(hpar['batch_size'])
end     = sum([v for _,v in env.timing.items()])/1000
times   = np.arange(end, step=env.dt/1000)
rules   = end - env.timing['decision']/1000, end

f, ax = plt.subplots(1,2,figsize=(16,5))
ax[0].axvspan(rules[0],rules[1], facecolor='grey', alpha=0.2)
ax[0].plot(times, inputs[:,i_trial,1], 'blue', label='Evidence for right motion')
ax[0].plot(times, inputs[:,i_trial,2], 'red', label='Evidence for left motion')
ax[0].plot(times, inputs[:,i_trial,0], 'green', label='Fixation rule')
ax[0].set_ylabel('input'); ax[0].set_ylim([-0.1,1.1])
ax[0].legend(loc='upper left')
ax[0].set_xlabel('time (s)')
ax[0].set_title(f'Input to RNN (trial number={i_trial})')

ax[1].axvspan(rules[0],rules[1], facecolor='grey', alpha=0.2)
ax[1].hlines(y=1, xmin=times[0], xmax=times[-1], color='gray', linestyle='dashed')
ax[1].hlines(y=2, xmin=times[0], xmax=times[-1], color='gray', linestyle='dashed')
ax[1].text(0,0.08, 'Fixation')
ax[1].text(0,0.9, 'Right decision')
ax[1].text(0,1.9, 'Left decision')
ax[1].plot(times, target[:,i_trial], 'k')
ax[1].set_xlabel('time (s)')
ax[1].set_yticks([0,1,2])
ax[1].set_ylabel('desired output'); ax[1].set_ylim([-0.1,2.1])
ax[1].set_title(f'Desired output from RNN (trial number={i_trial})')
plt.tight_layout()
plt.show()


# Architecture: Model definition
class RNN(torch.nn.Module):
	"""Recurrent network model.
	Inputs:
		input: tensor of shape (seq_len, batch, input_size)
		hidden: tensor of shape (batch, hidden_size), initial hidden activity
			if None, hidden is initialized through self.init_hidden()
	Outputs:
		output: tensor of shape (seq_len, batch, hidden_size)
		hidden: tensor of shape (batch, hidden_size), final hidden activity
	"""
	def __init__(self, input_size, output_size, **hpar):
		super().__init__()
		self.input_size  = input_size
		self.output_size = output_size
		self.hidden_size = hpar['n_neurons']
		self.noise       = hpar['network_noise']
		self.tau         = hpar['tau']
		self.alpha       = hpar['dt'] / self.tau
		self.input2rec   = torch.nn.Linear(self.input_size, self.hidden_size, bias=False)
		self.rec2rec     = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.rec2output  = torch.nn.Linear(self.hidden_size, self.output_size)
		self.normal      = torch.distributions.normal.Normal(0,1)
		self.set_activation_fun(**hpar)

	def set_activation_fun(self, **hpar):
		if hpar['activation_fun'] == 'sigmoid':
		  self.activation = torch.sigmoid
		elif hpar['activation_fun'] == 'relu':
		  self.activation = torch.relu 
		elif hpar['activation_fun'] == 'softplus':
		  self.activation = F.softplus
		else:
		  raise NotImplementedError('Activation functions should be either ReLU, Sigmoid, or Softplus')

	def init_hidden(self, input_shape):
		batch_size = input_shape[1]
		return torch.zeros(batch_size, self.hidden_size)

	def recurrence(self, input, hidden):
		"""Run network for one time step.
		Inputs:
			input: tensor of shape (batch, input_size)
			hidden: tensor of shape (batch, hidden_size)
		Outputs:
			h_new: tensor of shape (batch, hidden_size),
				network activity at the next time step
		"""
		noise = self.noise * self.normal.sample(hidden.shape)

		h_new = self.activation(self.input2rec(input) + self.rec2rec(hidden))
		h_new = hidden*(1-self.alpha) + h_new*self.alpha
		h_new = h_new + noise*(2*self.alpha)**0.5
		return h_new


	def forward(self, input, hidden=None):
		"""Propogate input through the network."""
		
		# If hidden activity is not provided, initialize it
		if hidden is None:
			hidden = self.init_hidden(input.shape).to(input.device)

		# Loop through time
		rnn_output = []
		steps  = range(input.size(0))
		for i in steps:
			hidden = self.recurrence(input[i], hidden)
			rnn_output.append(hidden)

		# Stack together output from all time steps
		rnn_output = torch.stack(rnn_output, dim=0)  # (seq_len, batch, hidden_size)

		# Return linear readout
		output = self.rec2output(rnn_output)

		return output, rnn_output

# Instantiate the network
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n
net         = RNN(input_size, output_size, **hpar)

print('RNN model trainable parameter shapes')
for name, param in net.named_parameters():
	if param.requires_grad:
		print ('\t', name, ':', param.data.shape)


# Model training
def train_model(net, dataset, **hpar):
	"""Simple helper function to train the model.
	Args:
		net: a pytorch nn.Module module
		dataset: a dataset object that when called produce a (input, target output) pair
	Returns:
		net: network object after training
	"""
	# Define an optimizer
	learning_rate = hpar['learning_rate'] 
	if   hpar['optimizer'] == 'Adam':
	  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
	elif hpar['optimizer'] == 'RMSprop':
	  optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
	elif hpar['optimizer'] == 'SGD':
	  optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
	else:
	  raise NotImplementedError('Optimizer should be either Adam, RMSprop, or SGD')

	# Define a loss function
	if   hpar['loss'] == 'CrossEntropy':
	  criterion = torch.nn.CrossEntropyLoss()
	elif hpar['loss'] == 'L1':
	  criterion = torch.nn.L1Loss()
	elif hpar['loss'] == 'L2':
	  criterion = torch.nn.MSELoss()
	else:
	  raise NotImplementedError('Loss function should be either CrossEntropy, L1, or L2')

	start_time   = time.time()
	losses       = []
	accs         = []

	# Loop over training batches
	for i in range(hpar['n_iteration']):
		# Generate input and target, convert to pytorch tensor
		inputs, labels = dataset()
		inputs = torch.from_numpy(inputs).type(torch.float)
		labels = torch.from_numpy(labels.flatten()).type(torch.long)

		# boiler plate pytorch training
		optimizer.zero_grad()   # clear the gradient buffers
		output, _ = net(inputs) # run the network
		output = output.view(-1, output_size) # reshape output

		# compute loss function
		if hpar['loss'] == 'CrossEntropy':
		  loss = criterion(output, labels)
		else:
		  loss = criterion(output, F.one_hot(labels).type(torch.float))
		loss.backward()  # backpropagation
		optimizer.step() # gradient descent

		# Compute the running loss every 100 steps
		losses.append(loss.item())
		pred = np.argmax(output.detach().numpy()[(-hpar['batch_size']):,:],axis=-1)
		true = labels.detach().numpy()[(-hpar['batch_size']):]
		accs.append((pred == true).mean()*100)
		if i % 100 == 99:
			print(f'Training loss={np.round(np.mean(losses[(-100):]),2)}')
			print(f'Training accuracy(%)={np.round(np.mean(accs[(-100):]),2)}')
			if i == hpar['n_iteration'] -1: print('Training finished.')
	return net

net = RNN(input_size, output_size, **hpar)
net = train_model(net, dataset, **hpar)


# Running RNN after training
def run_model(net, env, num_trial=200):
  """running a model after training"""
  env.reset(no_step=True)
  input_dict    = {}
  activity_dict = {} # recording activity
  trial_infos   = {} # recording trial information

  for i in range(num_trial):
	  trial_info = env.new_trial() # sample a new trial
	  ob, gt = env.ob, env.gt      # observation and groud-truth of this trial
	  inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
	  
	  # Run the network for one trial
	  action_pred, rnn_activity = net(inputs)
	  
	  # Compute performance
	  action_pred = action_pred.detach().numpy()[:, 0, :]
	  choice = np.argmax(action_pred[-1, :]) # final choice at last time step
	  correct = choice == gt[-1]             # compare to ground truth
	  
	  # Record activity, trial information, choice, correctness
	  _input = inputs[:, 0, :].detach().numpy()
	  rnn_activity = rnn_activity[:, 0, :].detach().numpy()
	  input_dict[i] = _input
	  activity_dict[i] = rnn_activity
	  trial_infos[i] = trial_info  # trial_info is a dictionary
	  trial_infos[i].update({'correct': correct, 'pred':action_pred, 'target': dataset.env.gt})
  
  return input_dict, activity_dict, trial_infos

## Running RNN
run_inputs, activity_dict, trial_infos = run_model(net, dataset.env)
test_acc = np.round(np.mean([val['correct'] for val in trial_infos.values()])*100,2)
print(f'Testing accuracy(%)={test_acc}')


# Randomly select one example trial (=input-output pair) from the task environment and plot the RNN behavior.
i_trial = np.random.choice(200)
end     = sum([v for _,v in env.timing.items()])/1000
times   = np.arange(end, step=env.dt/1000)
rules   = end - env.timing['decision']/1000, end

f, ax = plt.subplots(1,2,figsize=(16,5))
ax[0].axvspan(rules[0],rules[1], facecolor='grey', alpha=0.2)
ax[0].plot(times, run_inputs[i_trial][:,1], 'blue', label='Evidence for right motion')
ax[0].plot(times, run_inputs[i_trial][:,2], 'red', label='Evidence for left motion')
ax[0].plot(times, run_inputs[i_trial][:,0], 'green', label='Fixation rule')
ax[0].set_ylabel('input'); ax[0].set_ylim([-0.1,1.1])
ax[0].legend(loc='upper left')
ax[0].set_xlabel('time (s)')
ax[0].set_title(f'Input to RNN (trial number={i_trial})')

im = ax[1].imshow(softmax(trial_infos[i_trial]['pred'],axis=-1).T, extent=[times[0],end,-0.5,2.5], aspect='auto', origin='lower')
ax[1].text(0.05,0.1, 'Fixation', color='red')
ax[1].text(0.05,1.0, 'Right decision', color='red')
ax[1].text(0.05,2.0, 'Left decision', color='red')
ax[1].plot(times, trial_infos[i_trial]['target'], 'k', linewidth=5, label='ground truth')
ax[1].set_xlabel('time (s)')
ax[1].set_yticks([0,1,2])
ax[1].set_ylabel('RNN output'); ax[1].set_ylim([-0.5,2.5])
ax[1].set_title(f'RNN output (trial number={i_trial})')
ax[1].legend()
cb = plt.colorbar(im, ax=ax[1])
cb.ax.set_title('Predicted probability')
plt.tight_layout()
plt.show()


# Principal component analysis of RNN population activity
## Concatenate activity for PCA
activity = np.concatenate(list(activity_dict[i] for i in range(200)), axis=0)
pca = PCA(n_components=2)
pca.fit(activity)  # activity (Time points, Neurons)
activity_pc = pca.transform(activity)  # transform to low-dimension

## Project each trial and visualize activity
blues = cm.get_cmap('Blues', len(env.cohs)+1)
reds  = cm.get_cmap('Reds',  len(env.cohs)+1)

plt.figure(figsize=[7,7])
for i in range(100):
	# Transform and plot each trial
	activity_pc = pca.transform(activity_dict[i])  # (Time points, PCs)

	trial   = trial_infos[i]
	color_c = blues if trial['ground_truth'] == 0 else reds
	color   = color_c(np.where(env.cohs == trial['coh'])[0][0])
	plt.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)

	# Plot the beginning of a trial with a special symbol
	plt.plot(activity_pc[0, 0], activity_pc[0, 1], '^', color='black', markersize=10)

handles, labels = plt.gca().get_legend_handles_labels()

dot   = Line2D([], [], marker='^', label='Starting point', color='black', linestyle='None', markersize=10)
lines = []
for i_coh, v_coh in reversed(list(enumerate(env.cohs))):
  line_r = Line2D([], [], label=f'L motion trial, c={v_coh}', color=reds(i_coh)); lines.append(line_r)
for i_coh, v_coh in enumerate(env.cohs):
  line_b = Line2D([], [], label=f'R motion trial, c={v_coh}', color=blues(i_coh)); lines.append(line_b)

handles.extend([dot]+lines)
plt.legend(title='Legend', bbox_to_anchor=(1.4, 1.0), handles=handles, frameon=False)
plt.title('State space based on principal component analysis')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()