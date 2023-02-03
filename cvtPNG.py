import os
import time

import cv2
import glob
import gdcm
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from joblib import Parallel, delayed


def process(f, size=512, save_folder="", extension="png"):
    patient = f.split('/')[-2]
    image = f.split('/')[-1][:-4]

    dicom = pydicom.dcmread(f)
    img = dicom.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cv2.resize(img, (size, size))

    cv2.imwrite(save_folder + f"{patient}_{image}.{extension}", (img * 255).astype(np.uint8))


if __name__ == '__main__':
    train_images = glob.glob("../data/*/*.dcm")
    len(train_images)  # 54706

    SAVE_FOLDER = "../data/RSNABreastCancer/"
    SIZE = 512
    EXTENSION = "png"

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    _ = Parallel(n_jobs=4)(
        delayed(process)(uid, size=SIZE, save_folder=SAVE_FOLDER, extension=EXTENSION)
        for uid in tqdm(train_images))