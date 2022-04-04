import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import itertools
from scipy.sparse import csr_matrix

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	df = pd.read_csv(path)
	data_y = df[["y"]] - 1
	data_x = df.loc[:, df.columns != "y"]

	if model_type == 'MLP':
		data = torch.from_numpy(data_x.values).float()
		target = torch.from_numpy(data_y.values).squeeze(1).long()
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = torch.from_numpy(data_x.values).unsqueeze(1).float()
		target = torch.from_numpy(data_y.values).squeeze(1).long()
		dataset = TensorDataset(data, target)
	elif model_type == 'RNN':
		data = torch.from_numpy(data_x.values).unsqueeze(2).float()
		target = torch.from_numpy(data_y.values).squeeze(1).long()
		dataset = TensorDataset(data, target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	a =  list(itertools.chain.from_iterable(seqs))
	b = list(itertools.chain.from_iterable(a))     
	return max(b)+1


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		# sort input seqs by number of visits
		seqs.sort(key=len, reverse=True)
		max_rows = len(seqs[0])
		self.seqs = []
		for patient_data in seqs:
			row = 0
			data_vector = []
			row_vector = []
			col_vector = []
			for visit_data in patient_data:
				dv = list(np.ones(len(visit_data)))
				data_vector += dv
				rv = list(np.ones(len(visit_data)) * row)
				row_vector += rv
				cv = visit_data
				col_vector += cv
				row +=1
			mat = csr_matrix((data_vector, (row_vector, col_vector)), shape=(max_rows, num_features)).todense()
				
			self.seqs.append(mat)

		#self.seqs = [i for i in range(len(labels))]  # replace this with your implementation.

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence
	seqs = [torch.FloatTensor(x[0]) for x in batch]  
	labels = [x[1] for x in batch]  

	seqs_tensor = pad_sequence(seqs)
	lengths_tensor = torch.LongTensor([len(x) for x in batch])
	labels_tensor = torch.LongTensor(labels)

	return (seqs_tensor, lengths_tensor), labels_tensor
