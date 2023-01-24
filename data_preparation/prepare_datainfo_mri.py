#!/usr/bin/env python
# coding: utf-8

import os, glob
from pathlib import Path
import json
import numpy as np
import pandas as pd


root_dir = Path('/export/scratch3/grewal/Data/Projects_JPG_data/MO_DIR/MR_sag/MODIR_data_test_split')
csv_path = f'./meta/mri_dataset.csv'

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
        
        info_dict = {"root_path": root_dir,\
                    "path": path,\
                    "patient_id": Path(path).relative_to(root_dir).parts[0],\
                    "series_id": meta[0]["SeriesInstanceUID"],\
                    "series_date": meta[0]["SeriesDate"],\
                    "labels": labels,\
                    "no_of_contours": len(annotation),\
                    "applicator": "Yes"}
        info_list.append(info_dict)
    
df = pd.DataFrame(info_list)
# label_dummies = df.labels.apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0).astype(int)
# df = df.join(label_dummies)

df['labels'] = df['labels'].map(lambda x: "|".join(x))
df = df.groupby(by="patient_id").apply(lambda x: x.sort_values(by="series_date"))

"""
delete duplicate rows on [patient_id, series_date, labels, no_of_contours]
delete corresponding json file as well as jpg image
"""
duplicates_flag = df.duplicated(subset=["patient_id", "series_date", "labels", "no_of_contours"],\
                                keep="first")

df_duplicates = df.loc[duplicates_flag]
print(f"No. of duplicates: {len(df_duplicates)}")
for i, row in df_duplicates.iterrows():
    json_path = str(row.path)
    jpg_path = json_path.replace("split", "split_labels")
    jpg_path = jpg_path.replace(".json", ".jpg")

    os.remove(json_path)
    os.remove(jpg_path)

df_unique = df.loc[duplicates_flag==False]
df_unique.to_csv(csv_path, index=False)
