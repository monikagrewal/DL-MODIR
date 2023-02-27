#!/usr/bin/env python
# coding: utf-8

import os, glob
from pathlib import Path
import json
import numpy as np
import pandas as pd
import label_mapping


root_dir = Path('/export/scratch2/data/grewal/Data/Projects_JPG_data/MO_DIR/MR_sag/MODIR_data_train_split')
csv_path = f'./meta/mri_dataset_train_2.csv'

paths = list(root_dir.glob('**/*.json'))
print("total data: {}".format(len(paths)))

info_list = []
for path in paths:
    with open(str(path)) as f:
        info = json.loads(f.read())
        meta = info["meta"]
        annotation = info["annotation"]
        labels = [item["label_name"] for item in annotation]
        labels = np.unique(labels)
        labels = [label_mapping.map_label(label) for label in labels]
        labels = [label for label in labels if label is not None]
        labels = np.unique(labels)
        pixel_spacing = meta[0]['PixelSpacing'][:2]
        slice_thickness = meta[0]['SliceThickness']
        voxel_spacing = "|".join([str(pixel_spacing[0]), \
                                    str(pixel_spacing[1]), \
                                        str(slice_thickness)])
        
        info_dict = {"root_path": root_dir,\
                    "path": path,\
                    "patient_id": Path(path).relative_to(root_dir).parts[0],\
                    "series_id": meta[0]["SeriesInstanceUID"],\
                    "series_date": meta[0]["SeriesDate"],\
                    "voxel_spacing": voxel_spacing,\
                    "labels": labels,\
                    "no_of_contours": len(annotation),\
                    "applicator": "Yes"}
        info_list.append(info_dict)
    
df = pd.DataFrame(info_list)

# remove entries with no annotations
label_present = df.labels.apply(lambda x: len(x)!=0)
df = df[label_present]

# make 1/0 column for each label
label_dummies = df.labels.apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0).astype(int)
df = df.join(label_dummies)

df['labels'] = df['labels'].map(lambda x: "|".join(x))
df = df.groupby(by="patient_id").apply(lambda x: x.sort_values(by="series_date"))

df_unique = df.copy()
# """
# delete duplicate rows on [patient_id, series_date, labels, no_of_contours]
# delete corresponding json file as well as jpg image
# """
# duplicates_flag = df.duplicated(subset=["patient_id", "series_date", "labels", "no_of_contours"],\
#                                 keep="first")

# df_duplicates = df.loc[duplicates_flag]
# print(f"No. of duplicates: {len(df_duplicates)}")
# for i, row in df_duplicates.iterrows():
#     json_path = str(row.path)
#     jpg_path = json_path.replace("split", "split_labels")
#     jpg_path = jpg_path.replace(".json", ".jpg")

#     os.remove(json_path)
#     os.remove(jpg_path)

# df_unique = df.loc[duplicates_flag==False]

df_unique.to_csv(csv_path, index=False)
