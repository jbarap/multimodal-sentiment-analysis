# Simplified DETR: End-to-End Object Detection with Transformers

This is an adaptation of [detr-light][0] to predict facial keypoints instead of bounding
boxes. For additional information visit the original repository.

## Setup

Poetry:
- Run `poetry install`

Pip:
- Optionally create a virtualenv of your choice
- Note: The project uses a specific version of albumentations (albumentations==0.5.2), you may have
  issues with the library if you already had another version installed,
  use `pip3 install albumentations==0.5.2` before running the next step.
- Run `pip3 install -r requirements.txt`


## Usage

### Configurations

Training is configuration-based. Every config is a yaml file describing parameters of
training, model losses, matcher losses, datasets, etc. You may copy the base `flickr_faces.yaml`
configuration to customize your own. Configurations are stored under the `configs` folder.


### Datasets

The dataset used in this repo is a subset of [Flickr-Faces-HQ][1], created by the user
[prashantarorat][2] and posted on [Kaggle][3].

To facilitate the download of the data you may use the [Kaggle API][4]. To use it, follow
the instructions of their repo, which narrow down to:

- Install the python package with `pip3 install kaggle` (the package is already installed if using poetry)
- Go to the Account section of your Kaggle profile
- Select 'Create API Token', this will download a `kaggle.json` file that contains your token
- Move the `kaggle.json` file to the `~/.kaggle/` directory
- Check the functionality with `kaggle --version` or `poetry run kaggle --version`
- If you are using the API from Google Colab you can use the following snippet

```python
# from: https://colab.research.google.com/github/corrieann/kaggle/blob/master/kaggle_api_in_colab.ipynb
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# Then move kaggle.json into the folder where the API expects to find it.
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Once you have the API working, you can download and unzip the dataset with:

- `kaggle datasets download -d prashantarorat/facial-key-point-data -p data/facial_keypoints`
- `unzip data/facial_keypoints/facial-key-point-data.zip -d data/facial_keypoints/`


### Training

Training is done by calling `python -m detr.train`, use the `--help` flag to see options, but some of the possibilities
are: training just a section of the model, checkpoint interval, and config used.

### Inference

An inference script is provided for easier interface and visualization of the model. Inference on an image can
be performed by calling `python -m detr.inference` and providing the path to the model weights, the input image,
and the location under which the output image will be saved, use the `--help` flag for more details.



[0]: https://github.com/JA-Bar/detr-light
[1]: https://github.com/NVlabs/ffhq-dataset
[2]: https://www.kaggle.com/prashantarorat
[3]: https://www.kaggle.com/prashantarorat/facial-key-point-data
[4]: https://github.com/Kaggle/kaggle-api


