# -*- coding: utf-8 -*-
"""
Milestone 1 (Segmentation): Generation of an Annotated Dataset.
 1. Extract VOI (Volume of Interest) from the CTs (intensity and mask).
 2. Produce a single annotation for each lesion from the 4 radiologists’ annotations using Max-Voting.
 3. Make Max-Voting to obtain the “Diagnosis”: if two or more radiologists have characterized the nodule with a Malignancy score > 3, then Diagnosis=1 (malignant), otherwise Diagnosis=0 (benign).
"""

#%% Upload metadata
import pandas as pd

df = pd.read_excel('MetadatabyAnnotation.xlsx')
print(df.head())

df.columns

df[df['patient_id']=='LIDC-IDRI-0001']

#%% Extract VOIs

# load libraries
import nibabel as nib
import pandas as pd
import numpy as np
import os

def load_nii(filepath):
    """Load NIfTI image and return array and affine"""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

data1, affine1 =load_nii("/content/LIDC-IDRI-0001.nii.gz")

VOI_SIZE = 64
HALF_SIZE = VOI_SIZE // 2

def extract_voi(volume, center_voxel):
    """Extract a cubic VOI around the center voxel"""
    z, y, x = center_voxel
    return volume[z-HALF_SIZE:z+HALF_SIZE, y-HALF_SIZE:y+HALF_SIZE, x-HALF_SIZE:x+HALF_SIZE]

extract_voi(data1, )