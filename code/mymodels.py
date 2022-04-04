import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	# 3 layer mlp
	# hidden layer composed of 16 hidden units
	# sigmoid activation function
	def __init__(self):
		super(MyMLP, self).__init__()
		self.hidden1 = nn.Linear(178, 16)
		self.hidden2 = nn.Linear(16, 16)
		self.out = nn.Linear(16, 5)

	def forward(self, x):
		x = nn.functional.relu(self.hidden1(x))
		x = nn.functional.relu(self.hidden2(x))
		x = self.out(x)
		return x


class MyCNN(nn.Module):
	# 2 convolutional layers - 1) 6 filters of kernel size 5, stride 1, 2) 16 filters with kernel size 5 stride 1
	# relu activation
	# max pooling layer with size/stride of 2
	# two fully-connected layers, one with 128 hidden units followed by relu and other is output layer with 5 units
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding = 3)
		self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv1d(in_channels = 6, out_channels=16, kernel_size=5, stride=1, padding = 3)
		self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
		#self.fc1 = nn.Linear(in_features=16 * 178, out_features = 128)
		self.fc1 = nn.Linear(in_features=736, out_features = 128)
		self.fc2 = nn.Linear(in_features=128, out_features=5)

	def forward(self, x):
		x = self.pool1(nn.functional.relu(self.conv1(x)))
		x = self.pool2(nn.functional.relu(self.conv2(x)))
		#x = x.view(-1, 16 * 178)
		x = x.view(-1, 736)
		x = nn.functional.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class MyRNN(nn.Module):
	# gated recurrent unit w/ 16 hidden units followed by fully connected layer
	# many to one architecture
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=2, batch_first=True, dropout=0.5)
		self.fc = nn.Linear(in_features = 16, out_features = 5)

	def forward(self, x):
		x, _ = self.rnn(x)
		x = self.fc(x[:, -1, :])
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc = nn.Linear(in_features=dim_input, out_features=32)
		self.rnn = nn.GRU(input_size=32, hidden_size = 16, num_layers = 1, batch_first = True)
		self.fc2 = nn.Linear(in_features = 16, out_features = 2)


	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
		x = nn.functional.tanh(self.fc(input_tuple[0]))
		x, _ = self.rnn(x)
		x = self.fc2(x)
		return x