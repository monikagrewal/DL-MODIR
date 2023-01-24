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
	info = dict.fromkeys(["SeriesInstanceUID", "uid", "orientation",
			"origin", "SliceLocation", "PixelSpacing",
			 "SliceThickness", "Modality", "RescaleIntercept", "RescaleSlope", "PatientPosition",
			 "WindowWidth", "WindowCenter"], None)

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

		if attribute=="SOPInstanceUID":
			info["uid"] = im.SOPInstanceUID

		if attribute in ["SeriesInstanceUID", "PixelSpacing", "SliceThickness", "Modality",
			 "RescaleIntercept", "RescaleSlope", 
			 "PatientPosition", "WindowWidth", "WindowCenter"]:
			info[attribute] = eval('im.' + attribute)

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

def rescale_intensity(image, intercept, slope):
	if intercept is None or slope is None:
		return image

	# Convert to Hounsfield units (HU)
	image = np.float16(slope) * image.astype(np.float16) + np.float16(intercept)
	image = image.astype(np.int16)

	return image

def apply_ww_wl(image, ww, wl):
	ub = wl + ww//2
	lb = wl - ww//2
#     print(f"Upper bound: {ub}\nLower bound: {lb}")
	image[image > ub] = ub
	image[image < lb] = lb
	image = (image - lb) / float(ub - lb)
	return image

def normalize_array(image):
	image = (image - np.min(image)) / float(np.max(image) - np.min(image))
	return image

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


def resample_array(image, pixelspacing, fov=512, output_size=(512, 512)):
	shp = image.shape
	org_fov = round(shp[0] * float(pixelspacing[0]), 0)
	if org_fov < fov:
			pad = int((fov - org_fov)// (2 * float(pixelspacing[0])))
			image = np.pad(image, pad, mode='constant')
	elif org_fov > fov:
			crop = int((org_fov - fov)// (2 * float(pixelspacing[0])))
			image = image[crop : shp[0] - crop, crop : shp[1] - crop]

	image = resize(image, output_size, mode='constant')
	return image


def process_dicom_array(im, metadata):
	try:
		arr = im.pixel_array
	except Exception as e:
		print(f"Exception: {e}\n")
		return None

	# if arr.dtype==np.uint16:
	# 	print("The image data type is not readable for file: {}".format(str(pp)))
	# 	return None

	if arr.max() == arr.min():
		print("image is blank")
		return None

	if metadata["Modality"] == "CT":
		intercept = float(metadata["RescaleIntercept"])
		slope = float(metadata["RescaleSlope"])
		if isinstance(metadata["WindowWidth"], pydicom.multival.MultiValue):
			ww = float(metadata["WindowWidth"][0])
			wl = float(metadata["WindowCenter"][0])
		else:
			ww = float(metadata["WindowWidth"])
			wl = float(metadata["WindowCenter"])		
		arr = rescale_intensity(arr, intercept, slope)
		arr = apply_ww_wl(arr, ww, wl)
	arr = normalize_array(arr)
	if im.PatientPosition != "HFS":
		arr = arr[::-1, ::-1]

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
		try:
			acq_date = im.AcquisitionDate
			acq_time = im.AcquisitionTime
			try:
				acq_time = datetime.strptime(" ".join([acq_date, acq_time]), "%Y%m%d %H%M%S.%f")
			except:
				try:
					acq_time = datetime.strptime(" ".join([acq_date, acq_time]), "%Y%m%d %H%M%f")
				except:
					continue
			print(acq_time)
		except:
			continue
		# import pdb; pdb.set_trace()
	# 	metadata, status = extract_info(im)
	# 	if metadata['Modality'] == 'RTSTRUCT':
	# 		continue

	# 	if status and metadata["Modality"] == modality and metadata["orientation"] == orientation:
	# 		arr = process_dicom_array(im, metadata)
	# 		if arr is None:
	# 			continue
			
	# 		metadata['npixels'] = arr.shape

	# 		pp_rel = pp.relative_to(root_dir)
	# 		output_pp = (output_dir / pp_rel).with_suffix('.jpg')
	# 		output_pp.parent.mkdir(exist_ok=True, parents=True)
	# 		metadata['original_path'] = str(pp)
	# 		metadata['rel_path'] = str(pp_rel)

	# 		if output_directory is not None:
	# 			imsave(str(output_pp), (arr * 255).astype(np.uint8))
	# 			metadata['output_path'] = str(output_pp)
			
	# 		metadata = convert_dtypes(metadata)
	# 		series_id = metadata["SeriesInstanceUID"]
	# 		series_results = dicom_metadata.get(series_id, [])
	# 		series_results.append(metadata)
	# 		dicom_metadata[series_id] = series_results
	

	# for series_id, metadata_list in dicom_metadata.items():
	# 	with open(str(output_dir.parent / '{}.json'.format(series_id)), "w") as output_file:
	# 		json.dump(metadata_list, output_file)

	return None


def train_val_split(data, train_ratio=0.9, csv_path="./"):
	df = pd.DataFrame.from_dict({"path": data})

	np.random.seed(1234)
	indices = np.arange(len(data))
	num_train_ids = int(len(indices) * train_ratio)
	shuffled = np.random.permutation(indices)
	ids_train = shuffled[:num_train_ids]
	ids_val = shuffled[num_train_ids:]

	df = df.assign(train=df.index.isin(ids_train))
	df.to_csv(os.path.join(csv_path, "landmark_train_val_info.csv"))


def mark_cervical_patients(data, df_cervical, csv_path="./"):
	cervical_patients = df_cervical.loc[df_cervical.CervicalCancer==1, "PatientID"].to_list()
	df = pd.DataFrame.from_dict({"path": data})
	df["PatientID"] = df.path.apply(lambda x: os.path.split(x)[0].split("/")[-1])
	df = df.assign(test=df.PatientID.isin(cervical_patients))
	df["train"] = df.test.apply(lambda x: not x)
	df.to_csv(os.path.join(csv_path, "landmark_test_info.csv"))


if __name__ == '__main__':
	root_path = '/export/scratch2/grewal/Data/Projects_DICOM_data/ThreeD/MODIR_data_train_split'
	output_path = '/export/scratch3/grewal/Data/Projects_JPG_data/MO_DIR/CT/MODIR_data_train_split'

	root_dir = Path(root_path)
	output_dir = Path(output_path)

	for i, pp in enumerate(root_dir.glob('*/*')):
		print(f"\nProcessing {i} : {pp}\n")

		dicom_path = str(output_path / pp.relative_to(root_path))
		process_dicoms(str(pp), output_directory=dicom_path, modality="CT")


	# data = glob.glob(output_path + "*/*/*.json")
	# train_val_split(data, train_ratio=0.8, csv_path=output_path)
