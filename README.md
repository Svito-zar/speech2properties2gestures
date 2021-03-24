# Gesture Differently
Gesture differently: probabilistic speaker-independent gesture generation using normalizing flows.
Taras Kucherenko, Patrik Jonell, Rajmund Nagy, Michael Neff, Hedvig Kjellström and Gustav Eje Henter.

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
