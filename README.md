# Gesture Differently
*Speech2Properties2Gestures: Gesture-Property Prediction as a Tool for Generating Representational Gestures from Speech*.   
Taras Kucherenko, Rajmund Nagy, Michael Neff, Hedvig Kjellström and Gustav Eje Henter.

*Multimodal analysis of hand gesture properties predictibility*.    
Taras Kucherenko, Patrik Jonell, Rajmund Nagy, Michael Neff, Hedvig Kjellström and Gustav Eje Henter.

# Expected dataset folders
```
dataset/transcripts:
01_video.eaf       05_video.eaf   08_video.pfsx  13_video.eaf   19_video.eaf   24_video.eaf
01_video_NEW.pfsx  05_video.pfsx  09_video.eaf   14_video.eaf   19_video.pfsx  25_video.eaf
02_video.eaf       06_video.eaf   10_video.eaf   15_video.eaf   20_video.eaf   Corrected_V17K3.mov_enhanced.json
03_video.eaf       07_video.eaf   11_video.eaf   16_video.eaf   21_video.eaf
03_video.pfsx      07_video.pfsx  11_video.pfsx  16_video.pfsx  22_video.eaf
04_video.eaf       08_video.eaf   12_video.eaf   18_video.eaf   23_video.eaf
```
and
```
dataset/audio:
Corrected_V17K3.mov_enhanced.json  V16K3.mov_enhanced.wav   V21K3.mov_enhanced.wav  V4K3.mov_enhanced.wav
V10K3.mov_enhanced.wav             V17K3.mov_enhanced.json  V22K3.mov_enhanced.wav  V5K3.mov_enhanced.wav
V11K3.mov_enhanced.wav             V17K3.mov_enhanced.wav   V23K3.mov_enhanced.wav  V6K3.mov_enhanced.wav
V12K3.mov_enhanced.wav             V18K3.mov_enhanced.wav   V24K3.mov_enhanced.wav  V7K3.mov_enhanced.wav
V13K3.mov_enhanced.wav             V19K3.mov_enhanced.wav   V25K3.mov_enhanced.wav  V8K3.mov_enhanced.wav
V14K3.mov_enhanced.wav             V1K3.mov_enhanced.wav    V2K3.mov_enhanced.wav   V9K3.mov_enhanced.wav
V15K3.mov_enhanced.wav             V20K3.mov_enhanced.wav   V3K3.mov_enhanced.wav
```
# Running the code
```python
conda env create -f environment.yml
pip install -e .
cd my_code/data_processing/annotations/
python extract_binary_features.py  
python encode_text.py

# Note that he following warning is expected:
# "WARNING: Skipping recording 17 because of missing files: ['17_text.hdf5', '17_feat.hdf5']"
python create_dataset.py
python concatenate_gesture_properties.py 
python create_gesture_existance_array.py 
python remove_zeros.py

# Example runs
cd ../../predict_ges_existance/
python cross-validation.py prop_params/Speech2GestExist.yaml 
cd ../predict_ges_properites/
python cross-validation.py prop_params/main/Both_Phase.yaml 
```
_______________________________

# Data Processing
### To encode binary gesture properties
```
python my_code/data_processing/annotations/create_binary_gesture_labels.py
```
This will create 25 files (for each recording) of gesture binary features in `feat` subfolder of the `annotations` folder

### To encode text words into BERT word-embeddings
```
python my_code/data_processing/annotations/encode_text.py
```
This will create 25 files (for each recording) of text embeddings in `feat` subfolder of the `annotations` folder

### To split the dataset into Test and Train_n_Val
I always did this part manually by putting the corresponding files from `feat` subfolder here to `test` and `train_n_val` subfolders in the folder where I want to keep the data.

### To create the dataset files
```
python my_code/data_processing/annotations/create_dataset.py
```
This will create dataset files in the form `train_n_val_A_Semantic.npy` in the root folder of the same folder where `train_n_val` subfolder was placed.


# GestureFlow

### To start visualization server
```
gunicorn -b "0.0.0.0:5103" -w 2 app:app
```

### To train a model normally
```
python train.py hparams/no_autoregr_gpu.yaml -
```


### To do hyper-parameters search
```
python hparams_tuning.py hparams/no_autoregr_gpu.yaml -n 50
```

# Baselines

## Gesticulator 

https://github.com/nagyrajmund/gesticulator/tree/gestureflow_baseline

## StyleGestures

https://github.com/nagyrajmund/stylegestures/tree/baseline

## Trimodal (Yoon et al.)

https://github.com/nagyrajmund/Gesture-Generation-from-Trimodal-Context/tree/baseline
3
### NOTES
- The architecture of the audio CNN in the generator and the convolutional discriminator had to be changed to adapt the new motion dim.
- The autoencoder for calculating the FGD has not beed adapted yet
- For this model the hdf5 text dataset has been recreated to include word strings instead of the embeddings
