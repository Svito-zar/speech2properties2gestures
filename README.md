# Gesture Differently
*Speech2Properties2Gestures: Gesture-Property Prediction as a Tool for Generating Representational Gestures from Speech*.   
Taras Kucherenko, Rajmund Nagy, Michael Neff, Hedvig Kjellström and Gustav Eje Henter.
International Conference on Intelligent Virtual Agents 2021 (IVA’21) 

*Multimodal analysis of hand gesture properties predictibility*.    
Taras Kucherenko, Patrik Jonell, Rajmund Nagy, Michael Neff, Hedvig Kjellström and Gustav Eje Henter.
(to submit to) AAMAS 2022

# Dataset

You can download the data from [https://doi.org/10.5281/zenodo.6534502](https://zenodo.org/records/6546229).

## Expected dataset folders
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


# Setting up the environment
```python
conda env create -f environment.yml
conda activate speech2prop
pip install -e .
```

# Preprocessing the data
```python
cd my_code/data_processing/annotations/
python extract_binary_features.py  
python encode_text.py

# Note that he following warning is expected:
# "WARNING: Skipping recording 17 because of missing files: ['17_text.hdf5', '17_feat.hdf5']"
python create_dataset.py
python concatenate_gesture_properties.py 
python create_gesture_existance_array.py 
python remove_zeros.py
```

# Running the model
```python
# Example runs
cd ../../predict_ges_existance/
python cross-validation.py prop_params/Speech2GestExist.yaml 
cd ../predict_ges_properites/
python cross-validation.py prop_params/main/Both_Phase.yaml 
```

