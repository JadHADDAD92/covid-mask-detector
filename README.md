[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JadHADDAD92/covid-mask-detector/blob/master/LICENSE)
[![CodeQL](https://github.com/JadHADDAD92/covid-mask-detector/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/JadHADDAD92/covid-mask-detector/actions/workflows/codeql-analysis.yml)

<a href="https://www.buymeacoffee.com/jadhaddad" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" height="41" width="174">
</a>


# COVID-19 Face Mask Detector

![samples](images/testmask.gif)
# Reproduction
```Shell

git clone git@github.com:JadHADDAD92/covid-mask-detector.git
# Or
git clone https://github.com/JadHADDAD92/covid-mask-detector.git

cd covid-mask-detector

# Download dataset and export it to pandas DataFrame
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
#### Pretrained model
[covid-mask-detector/models/face_mask.ckpt](https://github.com/JadHADDAD92/covid-mask-detector/blob/master/covid-mask-detector/models/face_mask.ckpt)

**Check Medium post:** [How I built a Face Mask Detector for COVID-19 using PyTorch Lightning](https://towardsdatascience.com/how-i-built-a-face-mask-detector-for-covid-19-using-pytorch-lightning-67eb3752fd61)
