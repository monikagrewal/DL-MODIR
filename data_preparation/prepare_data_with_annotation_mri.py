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


def visualize_label(meta_list, annotations, label_pp, view="transverse"):
	# colors = {0: (1, 0, 0), 1: (1, 0, 1), 2: (0, 1, 0), 3: (0, 0, 1),
	# 			4: (1, 1, 0), 5: (0, 1, 1), 6: (1, 0, 1),
	# 			7: (1, 0.5, 0), 8: (0, 1, 0.5), 9: (0.5, 0, 1),
	# 			10: (0.5, 1, 0), 11: (0, 0.5, 1), 12: (1, 0, 0.5)}
	meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
	for i, meta in enumerate(meta_sorted):
		dicom_path = meta['original_path']
		im = pydicom.dcmread(dicom_path)
		img = process_dicom_array(im, meta)

		img = np.stack((img,)*3, axis=-1)
		combined = img.copy()
		opacity = 0.8
		uid = meta['uid']
		annotation = [item for item in annotations if item["uid"]==uid]
		for idx, item in enumerate(annotation):
			coords = item["coords"]
			coords = np.asarray(coords).squeeze()
			if coords.ndim==1:
				coords = coords.astype(np.int16)
				rr, cc = skimage.draw.disk((coords[1], coords[0]), radius=2, shape=img.shape)
				combined[rr, cc] = np.array((1, 0, 0))
			else:
				pass
				# rr, cc = skimage.draw.polygon(coords[:,1], coords[:,0], shape=img.shape)
				# combined[rr, cc] = opacity*np.array(item["color"]) + (1-opacity)*combined[rr, cc]
		
		combined = np.concatenate((combined, img), axis=1)
		output_path = str(label_pp) + f"/{i}.jpg"
		imsave(output_path, (combined*255).astype(np.uint8))



def visualize_series(meta_list, filepath, view="sagittal"):
	meta_sorted = sorted(meta_list, key=lambda x: x['SliceLocation'])
	img_array = []
	for i, meta in enumerate(meta_sorted):
		dicom_path = meta['original_path']
		im = pydicom.dcmread(dicom_path)
		img = process_dicom_array(im, meta)
		img_array.append(img)
	
	img_array = np.array(img_array, dtype=np.float32) #cc * ap * rl

	# interpolation to make equal spacing
	pixel_spacing = float(meta_sorted[0]["PixelSpacing"][0])
	slice_thickness = meta_sorted[0]["SliceThickness"]
	# img_array = scipy.ndimage.zoom(img_array, (np.round(slice_thickness/pixel_spacing),\
	# 								 1, 1), order=1)
	
	imlist = []
	if view=="transverse":
		slices = img_array.shape[0]
		mid_slices = [slices//2 - 8, slices//2 + 8]
		for i in range(mid_slices[0], mid_slices[1]):
			img = img_array[i, :, :]
			imlist.append(img)
	elif view=="sagittal":
		slices = img_array.shape[1]
		mid_slices = [slices//2 - 8, slices//2 + 8]
		for i in range(mid_slices[0], mid_slices[1]):
			img = img_array[:, :, i]
			imlist.append(img)
	elif view=="coronal":
		slices = img_array.shape[2]
		mid_slices = [slices//2 - 8, slices//2 + 8]
		for i in range(mid_slices[0], mid_slices[1]):
			img = img_array[:, i, :]
			imlist.append(img)
	else:
		raise ValueError("Unknown view: {view}. Supported views are: transverse, coronal, sagittal")

	del img_array
	new_imlist = []
	for i in [0, 4, 8, 12]:
		horizontal_im = np.concatenate((imlist[i],\
										imlist[i+1],\
										imlist[i+2],\
										imlist[i+3]), axis=1)

		new_imlist.append(horizontal_im)
	full_im = np.concatenate(new_imlist, axis=0)
	imsave(filepath, (full_im*255).astype(np.uint8))


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

		if attribute=="PixelSpacing":
			info["PixelSpacing"] = [float(item) for item in im.PixelSpacing]

		if attribute=="SOPInstanceUID":
			info["uid"] = im.SOPInstanceUID

		if attribute in ["SeriesInstanceUID", "PixelSpacing", "SliceThickness", "Modality",
			 "RescaleIntercept", "RescaleSlope", 
			 "PatientPosition", "WindowWidth", "WindowCenter", "SeriesDate"]:
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
	return image.astype(np.float32)


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


def process_annotation(item, meta):
	meta = meta[0]
	patientposition = meta['PatientPosition']
	origin = np.array(meta['origin'][:2]).reshape(1,2)
	pixelspacing = np.array(meta['PixelSpacing'][:2]).reshape(1,2)

	coords = item["coords"]
	# print(np.array(coords).shape, meta["origin"])
	if patientposition=="HFP":
		orientation = np.array([-1, -1])
		coords_pix = orientation*np.array(coords) - orientation*origin	
		coords_pix = coords_pix / pixelspacing
		coords_pix = meta['npixels'] - coords_pix
	else:
		coords_pix = np.array(coords) - origin
		coords_pix = coords_pix / pixelspacing

	item["coords"] = coords_pix.tolist()

	return item


def grouper(iterable, n, fillvalue=None):
	"Collect data into fixed-length chunks or blocks"
	# grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
	args = [iter(iterable)] * n
	return zip_longest(fillvalue=fillvalue, *args)


def process_rtstruct(rtstruct):
	annotation = []
	# include = ['rectum', 'hip', 'bowel', 'bladder', 'sigmoid', 'spinal', 'anal_canal', 'anal canal', 'blaas']
	# exclude = ['ctv','ptv','gtv', 'hippo']
	for label_idx, roi in enumerate(rtstruct.ROIContourSequence):
		try:
			label_name = rtstruct.StructureSetROISequence[label_idx].ROIName.lower()
			color = list(roi.ROIDisplayColor)
			color = [int(item) for item in color]

			for cont in roi.ContourSequence:
				if cont.ContourGeometricType == 'POINT':
					# print("Annotation is a point")
					coords = np.array(list(cont.ContourData)).reshape(-1, 3)[:, 0:2]
					# assert coords.shape==(1, 3)
				elif cont.ContourGeometricType == 'CLOSED_PLANAR':
					# print("Annotation is contour")
					coords = np.array(list(grouper(cont.ContourData, 3)))[:, 0:2]
				else:
					print("Unexpected geometric type: ", cont.ContourGeometricType)
					continue
				
				if hasattr(cont, "ContourImageSequence"):
					uid = cont.ContourImageSequence[0].ReferencedSOPInstanceUID
				else:
					uid = ""

				entry = {"uid": uid,
						"label_name": label_name,
						"coords": coords.tolist(),
						"color": tuple(color)}
				annotation.append(entry)
		except Exception as e:
			print(str(e))
			continue

	return annotation


def match_dicoms_and_annotation(dicom_metadata, annotations):
	series_info = {}
	for series_id, metadata_list in dicom_metadata.items():
		series_info[series_id] = {"meta": metadata_list}
		nmatch = 0
		for _, annotation in annotations.items():
			annot_uids = [item["uid"] for item in annotation]
			matching_uids = [meta["uid"] for meta in metadata_list if meta["uid"] in annot_uids]
			if len(matching_uids) > 1:
				annotation = list(map(lambda x: process_annotation(x, metadata_list), annotation))
				existing_match = series_info[series_id].get("annotation", None)
				if existing_match is None:
					series_info[series_id]["annotation"] = annotation
				else:
					keyname = f"annotation_{nmatch}"
					series_info[series_id][keyname] = annotation
				
				nmatch += 1

	return series_info


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
	if metadata["PatientPosition"] == "HFP":
		arr = arr[::-1, ::-1]

	return arr


def process_dicoms(input_directory, output_directory, save_jpg=False, \
	label_output_dir=None, orientation="Transverse", modality="MR"):
	"""
	args:
	  input_directory: path to study date directory
	"""
	
	root_dir  = Path(input_directory)
	output_dir  = Path(output_directory)
	output_dir.mkdir(exist_ok=True, parents=True)
	dicom_metadata = {}
	annotations = {}
	for i, pp in enumerate(root_dir.glob('**/*.dcm')):
		if not pp.is_file():
			continue
		# import pdb; pdb.set_trace()        
		im = pydicom.dcmread(str(pp))
		metadata, status = extract_info(im)
		if metadata['Modality'] == 'RTSTRUCT':
			annotations[pp] = process_rtstruct(im)
			continue

		if metadata["Modality"] == modality and metadata["orientation"] == orientation:
			arr = process_dicom_array(im, metadata)
			if arr is None:
				continue
			
			metadata['npixels'] = arr.shape

			pp_rel = pp.relative_to(root_dir)
			metadata['original_path'] = str(pp)
			metadata['rel_path'] = str(pp_rel)

			metadata = convert_dtypes(metadata)
			series_id = metadata["SeriesInstanceUID"]
			series_results = dicom_metadata.get(series_id, [])
			series_results.append(metadata)
			dicom_metadata[series_id] = series_results

			output_pp = (output_dir / pp_rel).with_suffix('.jpg')
			if save_jpg:
				imsave(str(output_pp), (arr * 255).astype(np.uint8))
				metadata['output_path'] = str(output_pp)

	
	series_info = match_dicoms_and_annotation(dicom_metadata, annotations)

	if len(series_info)>0 and label_output_dir is not None:
		label_output_dir = Path(label_output_dir)
		label_output_dir.mkdir(exist_ok=True, parents=True)
	for series_id, info in series_info.items():
		# last match overwrites all preceding matches. assuming that most of the studies have only one match
		with open(str(output_dir / f'{series_id}.json'), "w") as output_file:
			json.dump(info, output_file)

		if label_output_dir is not None:
			label_pp = (label_output_dir / series_id)
			label_pp.mkdir(exist_ok=True, parents=True)
			visualize_label(info["meta"], info["annotation"], label_pp, view="transverse")
			filepath = os.path.join(str(label_output_dir), f"{series_id}.jpg")
			visualize_series(info["meta"], filepath, view="transverse")
			
	return None


if __name__ == '__main__':
	# root_path = '/export/scratch2/data/grewal/Data/Projects_DICOM_data/LUMC_cervical/train'
	# output_path = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_train'
	# label_output_path = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_train_labels'

	# root_path = '/export/scratch2/data/grewal/Data/Projects_DICOM_data/ThreeD/MRI_brachy_landmarks_annotations'
	# output_path = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/AMC_cervical_test_annotated'
	# label_output_path = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/AMC_cervical_test_annotated_labels'

	root_path = '/export/scratch2/data/grewal/Data/Projects_DICOM_data/LUMC_cervical/annotated'
	output_path = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_test_annotated'
	label_output_path = '/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/LUMC_cervical_test_annotated_labels'
	
	root_dir = Path(root_path)
	output_dir = Path(output_path)
	label_output_dir = Path(label_output_path)

	for i, pp in enumerate(root_dir.glob('*')):
		# if str(pp) not in ["/export/scratch2/data/grewal/Data/Projects_DICOM_data/LUMC_cervical/train/CervixRT323",
		#      				"/export/scratch2/data/grewal/Data/Projects_DICOM_data/LUMC_cervical/train/CervixRT331",
		# 				     "/export/scratch2/data/grewal/Data/Projects_DICOM_data/LUMC_cervical/train/CervixRT337",
		# 				     "/export/scratch2/data/grewal/Data/Projects_DICOM_data/LUMC_cervical/train/CervixRT380"
		# 					]:
		# 	continue
		print(f"\nProcessing {i} : {pp}\n")
		# if i >= 1:
		# 	break

		dicom_path = str(output_path / pp.relative_to(root_path))
		dicom_label_path = str(label_output_path / pp.relative_to(root_path))
		process_dicoms(str(pp), dicom_path, label_output_dir=dicom_label_path, orientation="Transverse")


