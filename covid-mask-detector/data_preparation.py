""" Add images into a pandas Dataframe
"""
from pathlib import Path
import cv2
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm

# download dataset from link provided by
# https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
datasetPath = Path('./dataset/mask.zip')
gdd.download_file_from_google_drive(file_id='1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp',
                                    dest_path=str(datasetPath),
                                    unzip=True)
# delete zip file
datasetPath.unlink()

datasetPath = Path('dataset/self-built-masked-face-recognition-dataset')
maskPath = datasetPath/'AFDB_masked_face_dataset'
nonMaskPath = datasetPath/'AFDB_face_dataset'
maskDF = pd.DataFrame()

for subject in tqdm(list(maskPath.iterdir()), desc='mask photos'):
    for imgPath in subject.iterdir():
        image = cv2.imread(str(imgPath))
        maskDF = maskDF.append({
            'image': image,
            'mask': True
        }, ignore_index=True)

for subject in tqdm(list(nonMaskPath.iterdir()), desc='non mask photos'):
    for imgPath in subject.iterdir():
        image = cv2.imread(str(imgPath))
        maskDF = maskDF.append({
            'image': image,
            'mask': False
        }, ignore_index=True)

dfName = 'data/mask_df.pickle'
print(f'saving Dataframe to: {dfName}')
maskDF.to_pickle(dfName)
