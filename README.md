# Segmentation of Multiple Sclerosis (MS) lesions using 3D-Unet

This project is testing the usage of 3D-Unet in segmenting brain lesions of patients with Multiple Sclerosis. It is a part of research for Master Thesis focused on enhancing 2D segmentation of MS lesions with deep neural network. The technical implementation, datasets, as well as results can be found [here](https://dspace.vutbr.cz/handle/11012/196900?locale-attribute=en).

## How to start

1. Prepare your dataset - train, validation and test dataset.
2. Load it and compress with `data_preprocess2021mar_3D.ipynb`.
3. Train your model with `server_code_model2021.py`.
4. Predict on test data.


## Contents of each file:

#### Dataset preprocessing
`data_preprocess2021mar_3D.ipynb` - Loading and preprocessing input data.

#### Compression
`compression.ipynb` - Visualisation of compression, which is being used during data loading to decrease the 3D picture size.

#### Training + Predicting
`server_code_model2021.py` - File to be run on server (python3). Includes implementation of 3D U-Net model created by Keras.



