import json
import os
import glob
from pathlib import Path
import pydicom
import numpy as np
import scipy
import skimage
from skimage.io import imread, imsave
from skimage.transform import resize
import pickle
from collections import Counter
from itertools import zip_longest
import shutil
import pandas as pd

def normalize_array(image:np.array):
	image = (image - np.min(image)) / float(np.max(image) - np.min(image))
	return image.astype(np.float32)


def visualize_series(meta_list:list, output_dir:str):
	"""
	TODO: visualize all slices instead of a few
	implemented only sagittal
	"""
	meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
	img_array = []
	for i, meta in enumerate(meta_sorted):
		dicom_path = meta['original_path']
		im = pydicom.dcmread(dicom_path)
		img = normalize_array(im.pixel_array)
		img_array.append(img)
	
	img_array = np.array(img_array, dtype=np.float32) #cc * ap * rl

	# interpolation to make equal spacing
	pixel_spacing = float(meta_sorted[0]["PixelSpacing"][0])
	slice_thickness = meta_sorted[0]["SliceThickness"]
	img_array = scipy.ndimage.zoom(img_array, (np.round(slice_thickness/pixel_spacing),\
									 1, 1), order=1)

	slices = img_array.shape[1]
	for i in range(0, slices):
		img = img_array[:, :, i]
		filepath = os.path.join(output_dir, f"{i}.jpg")
		imsave(filepath, (img*255).astype(np.uint8))


csv_path = f'./meta/mri_dataset_applicator_annotation.csv'
df = pd.read_csv(csv_path)
# df = df[df["applicator"]=="recheck"]

for i, row in df.iterrows():
	print(i)
	if row.patient_id=="1069510686_3358620618" and \
		row.series_id=="1.2.826.0.1.3680043.2.135.737157.54716028.7.1559620027.312.73":
		json_path = row.path
		with open(str(json_path)) as f:
			info = json.loads(f.read())
			meta = info["meta"]

		# make output directory
		# call visualize series in output directory
		output_dir = json_path.replace(".json", "")
		os.makedirs(output_dir, exist_ok=True)
		visualize_series(meta, output_dir)