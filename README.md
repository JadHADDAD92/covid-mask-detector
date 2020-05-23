# COVID-19 Face Mask Detector

![samples](images/testmask.gif)
# Reproduction
```Shell
git clone git@github.com:JadHADDAD92/covid-mask-detector.git
cd covid-mask-detector
python -m covid-mask-detector.data_preparation
```
## Training

```Shell
python -m covid-mask-detector.train
```

## Testing on videos
```sh
python -m covid-mask-detector.video modelPath videoPath
```

### Usage
```
Usage: video.py [OPTIONS] MODELPATH VIDEOPATH

  modelPath: path to model.ckpt

  videoPath: path to video file to annotate

Options:
  --output PATH  specify output path to save video with annotations
```

**Check Medium post:** [How I built a Face Mask Detector for COVID-19 using PyTorch Lightning](https://towardsdatascience.com/how-i-built-a-face-mask-detector-for-covid-19-using-pytorch-lightning-67eb3752fd61)
