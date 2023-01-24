import json
import os
import glob
from pathlib import Path
import pydicom
import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.transform import resize
import pickle
from collections import Counter
from itertools import zip_longest
import shutil
import pandas as pd
from datetime import datetime


def extract_info(im):
	info = dict.fromkeys(["SeriesInstanceUID", "SOPInstanceUID", "orientation",
			"origin", "SliceLocation", "PixelSpacing",
			 "SliceThickness", "Modality", "RescaleIntercept", "RescaleSlope", "PatientPosition",
			 "WindowWidth", "WindowCenter"], None)
	info["AcquisitionTime"] = "unspecified"

	for attribute in im.dir():
		if attribute=="ImageOrientationPatient":
			IOP = im.ImageOrientationPatient
			orientation = get_file_plane(IOP)
			info["orientation"] = orientation

		if attribute=="ImagePositionPatient":
			origin = im.ImagePositionPatient
			origin = [float(item) for item in origin]
			info["origin"] = origin

		if attribute=="SliceLocation":
			info["SliceLocation"] = float(im.SliceLocation)

		if attribute in ["SOPInstanceUID", "SeriesInstanceUID", "PixelSpacing", "SliceThickness", "Modality",
			 "RescaleIntercept", "RescaleSlope", 
			 "PatientPosition", "WindowWidth", "WindowCenter"]:
			info[attribute] = eval('im.' + attribute)

		if attribute=="AcquisitionTime":
			acq_date = im.AcquisitionDate
			acq_time = im.AcquisitionTime
			info[attribute] = " ".join([acq_date, acq_time])

	status = True
	for _, val in info.items():
		if val is None:
			status = False
			break

	return info, status

def get_file_plane(IOP):
	"""
	This function takes IOP of an image and returns its plane (Sagittal, Coronal, Transverse)

	Usage:
	a = pydicom.read_file(filepath)
	IOP = a.ImageOrientationPatient
	plane = file_plane(IOP)

	"""
	IOP_round = [round(x) for x in IOP]
	plane = np.cross(IOP_round[0:3], IOP_round[3:6])
	plane = [abs(x) for x in plane]
	if plane[0] == 1:
		return "Sagittal"
	elif plane[1] == 1:
		return "Coronal"
	elif plane[2] == 1:
		return "Transverse"


def convert_dtypes(metadata):
	dtype_mapping = {
		'MultiValue': list,
		'DSfloat': float
	}
	for k,v in metadata.items():
		dtype = v.__class__.__name__
		if dtype in dtype_mapping:
			metadata[k] = dtype_mapping[dtype](metadata[k])
	return metadata


def check_dicom_array(im):
	try:
		arr = im.pixel_array
	except Exception as e:
		print(f"Exception: {e}\n")
		return None

	if arr.max() == arr.min():
		print("image is blank")
		return None

	return arr



def process_dicoms(input_directory, output_directory=None, orientation="Transverse", modality="CT"):
	"""
	args:
	  input_directory: path to study date directory
	"""
	
	root_dir  = Path(input_directory)
	output_dir  = Path(output_directory)
	dicom_metadata = {}
	for i, pp in enumerate(root_dir.glob('**/*.dcm')):
		if not pp.is_file():
			continue        
		im = pydicom.dcmread(str(pp))
		metadata, status = extract_info(im)

		if status and metadata["Modality"] == modality and metadata["orientation"] == orientation:
			arr = check_dicom_array(im)
			if arr is None:
				continue
			
			metadata['npixels'] = arr.shape

			pp_rel = pp.relative_to(root_dir)
			output_pp = (output_dir / pp_rel).with_suffix('.jpg')
			output_pp.parent.mkdir(exist_ok=True, parents=True)
			metadata['original_path'] = str(pp)
			metadata['rel_path'] = str(pp_rel)
			metadata["PatientID"] = str(output_pp.parent).split("/")[-1]

			metadata = convert_dtypes(metadata)
			series_id = metadata["SeriesInstanceUID"]
			series_results = dicom_metadata.get(series_id, [])
			series_results.append(metadata)
			dicom_metadata[series_id] = series_results
	
	for series_id, metadata_list in dicom_metadata.items():
		with open(str(output_dir.parent / '{}.json'.format(series_id)), "w") as output_file:
			json.dump(metadata_list, output_file)

	return None


def create_raw_data_info(root_path, data, csv_path="./"):
	df = pd.DataFrame.from_dict({"path": data})
	PatientIDs = []
	acq_times = []
	for json_path in data:
		PatientID = os.path.split(json_path)[0].split("/")[-1]
		PatientIDs.append(PatientID)

		series_info = json.load(open(json_path, "r"))
		acq_time = series_info[0]["AcquisitionTime"]
		acq_times.append(acq_time)

	df["PatientID"] = PatientIDs
	df["AcquisitionTime"] = acq_times

	"""
	discard some patientIDs based on:
	1. the scan belongs to phantom/brain/blank etc.
	2. patient age range outside 18-95, based on info from Jan
	"""
	to_discard = json.load(open("/export/scratch3/grewal/Data/discarded_series_ids.json", "r"))
	df["series_json_path"] = df.path.apply(lambda x: x.replace(root_path, ""))
	df = df[~df["series_json_path"].isin(to_discard)]
	df = df.drop("series_json_path", axis=1)

	df.to_csv(os.path.join(csv_path, "data_info_raw.csv"))


def create_fixed_moving_pairs(raw_data_csv_path):
	df = pd.read_csv(raw_data_csv_path)
	grouped = df.groupby("PatientID")
	pairs = {"path_fixed": [], "path_moving": []}

	# # # --- method 1 ---
	# for name, group in grouped:
	# 	paths = group["path"].tolist()
	# 	while len(paths)>1:
	# 		for i in range(1, len(paths)):
	# 			pairs["path_fixed"].append(paths[0])
	# 			pairs["path_moving"].append(paths[i])
	# 		paths.pop(0)

	# # --- method 2 ---
	for name, group in grouped:
		paths = group["path"].tolist()
		if len(paths)>1:
			pairs["path_fixed"].append(paths[0])
			pairs["path_moving"].append(paths[1])

	df_new = pd.DataFrame.from_dict(pairs)
	csv_path = os.path.split(raw_data_csv_path)[0]
	df_new.to_csv(os.path.join(csv_path, "data_info_pairs_subset.csv"))


if __name__ == '__main__':
	root_path = '/export/scratch2/grewal/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split'
	output_path = '/export/scratch3/grewal/Data/Projects_JPG_data/MO_DIR/CT/MODIR_data_train_split'

	# root_dir = Path(root_path)
	# output_dir = Path(output_path)

	# for i, pp in enumerate(root_dir.glob('*/*')):
	# 	print(f"\nProcessing {i} : {pp}\n")

	# 	dicom_path = str(output_path / pp.relative_to(root_path))
	# 	process_dicoms(str(pp), output_directory=dicom_path, modality="CT")


	# data = glob.glob(output_path + "*/*/*.json")
	# create_raw_data_info(output_path, data, csv_path=output_path)
	create_fixed_moving_pairs(os.path.join(output_path, "data_info_raw.csv"))
	

	# filepath = "/export/scratch3/grewal/Data/Projects_JPG_data/ThreeD/CT/MODIR_data_train_split/landmarks_train_val_info_merged.csv"
	# root_dir = "/export/scratch3/grewal/Data/Projects_JPG_data/ThreeD/CT/MODIR_data_train_split/"
	# df = pd.read_csv(filepath)
	# df["series_json_path"] = df.path.apply(lambda x: x.replace(root_dir, ""))
	# out = df["series_json_path"].tolist()
	# json.dump(out, open("/export/scratch3/grewal/Data/discarded_series_ids.json", "w"))
	# print(out)


	# 	try:
	# 	info[attribute] = datetime.strptime(" ".join([acq_date, acq_time]), "%Y%m%d %H%M%S.%f")
	# except:
	# 	try:
	# 		info[attribute] = datetime.strptime(" ".join([acq_date, acq_time]), "%Y%m%d %H%M%f")
	# 	except:
	# 		continue