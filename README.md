# Instance Segmentation on Cityscapes dataset

Instance segmentation on the cityscapes dataset.  
Project for computer vision classes at Poznań University of Technology.

## Data
### Cityscapes
The cityscapes dataset can be downloaded by running ```data/scripts/get_files.sh```.  
### DVC
The model weights, tensorboard training logs and the custom dataset can be pulled with DVC:
 - Download service account login credentials from us.
 - Navigate to the ```data``` folder
 - Run ```dvc remote modify --local gcp_bucket credentialpath 'PATH/TO/CREDENTIALS'``` to authorize dvc to pull from google cloud.
 - Run ```dvc pull``` to download the data.

## Docker environment

Build
```
docker build -t cv3 .
```

Run
```
docker run \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v ./data:/app/CV_project3/data \
-p 8501:8501 \
-it --rm cv3
```

Run with CUDA
```
docker run \
--gpus all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v ./data:/app/CV_project3/data \
-p 8501:8501 \
-it --rm cv3
```

## Goals table:

| **Task** | **Points** | **Done** |
| --- | --- | --- |
| **Problem** |
| Instance Segmentation | 3 | X |
| Additional loss functions to improve prediction quality | 1 | X |
| **Model** |
| Our own model| 2 | X |
| architecture from the internet trained from scratch (mask_rcnn) | 1 | X |
| **Dataset** |
| Your own part of the dataset (>500 photos) | 1 | X |
| **Training** |
| Data augmentation | 1 | X |
| Testing a few optimizers | 1 | X |
| **Additional points** |
| Tensorboard | 1 | X |
| Docker | 1 | X |
| Streamlit | 1 | X |
| DVC | 2 | X |
| **Total** | **15** | **15** |
