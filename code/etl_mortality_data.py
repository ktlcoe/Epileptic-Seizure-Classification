import os
import pickle
import pandas as pd

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"


def convert_icd9(icd9_object):
	"""
	:param icd9_object: ICD-9 code (Pandas/Numpy object).
	:return: extracted main digits of ICD-9 code
	"""
	icd9_str = str(icd9_object)
	if icd9_str[0]=="E":
		converted = icd9_str[0:4]
	else:
		converted = icd9_str[0:3]
	# TODO: Extract the the first 3 or 4 alphanumeric digits prior to the decimal point from a given ICD-9 code.
	# TODO: Read the homework description carefully.
	return converted


def build_codemap(df_icd9, transform):
	"""
	:return: Dict of code map {main-digits of ICD9: unique feature ID}
	"""
	# TODO: We build a code map using ONLY train data. Think about how to construct validation/test sets using this.
	df_icd9 = df_icd9[~df_icd9["ICD9_CODE"].isnull()]   
	df_digits = df_icd9['ICD9_CODE'].apply(transform).unique()
	my_dict = dict(enumerate(df_digits))
	codemap = dict((v,k) for k,v in my_dict.items()) 
	return codemap


def create_dataset(path, codemap, transform):
	"""
	:param path: path to the directory contains raw files.
	:param codemap: 3-digit ICD-9 code feature map
	:param transform: e.g. convert_icd9
	:return: List(patient IDs), List(labels), Visit sequence data as a List of List of List.
	"""
	# TODO: 1. Load data from the three csv files
	# TODO: Loading the mortality file is shown as an example below. Load two other files also.
	df_mortality = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
	df_admission = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
	df_diagnosis = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))

	df_admission = df_admission.sort_values(by=["ADMITTIME"])
	df_diagnosis = df_diagnosis.sort_values(by=["SEQ_NUM"])

	# TODO: 2. Convert diagnosis code in to unique feature ID.
	# TODO: HINT - use 'transform(convert_icd9)' you implemented and 'codemap'.
	# drop rows that do not exist in codemap
	df_diagnosis = df_diagnosis[~df_diagnosis["ICD9_CODE"].isnull()]
	df_diagnosis["icd9_converted"] = df_diagnosis["ICD9_CODE"].transform(convert_icd9)
	df_diagnosis = df_diagnosis[df_diagnosis["icd9_converted"].isin(codemap.keys())]
	df_diagnosis["feature_id"] = df_diagnosis["icd9_converted"].apply(lambda x: codemap[x])
	visit_list = set(df_diagnosis["HADM_ID"].values.tolist())

	# TODO: 3. Group the diagnosis codes for the same visit.
	diag_groups = df_diagnosis.groupby("HADM_ID")["feature_id"].apply(list)

	# TODO: 4. Group the visits for the same patient.
	df_admission = df_admission[df_admission["HADM_ID"].isin(visit_list)]
	visit_groups = df_admission.groupby("SUBJECT_ID")["HADM_ID"].apply(list)

	# TODO: 5. Make a visit sequence dataset as a List of patient Lists of visit Lists
	# TODO: Visits for each patient must be sorted in chronological order.
	seq_data = []
	for i in visit_groups:
		tmp = []
		for j in i:
			diags = diag_groups[j]
			tmp.append(diags)
		seq_data.append(tmp)
			

	# TODO: 6. Make patient-id List and label List also.
	# TODO: The order of patients in the three List output must be consistent.
	patient_ids = df_mortality["SUBJECT_ID"].values.tolist()
	labels = df_mortality["MORTALITY"].values.tolist()
	#seq_data = [[[0, 1], [2]], [[1, 3, 4], [2, 5]], [[3], [5]]]
	return patient_ids, labels, seq_data


def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
