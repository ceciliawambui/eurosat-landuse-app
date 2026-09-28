# EuroSAT land-use classifier

A small convolutional neural network (model M2 in my CN7023 report) that sorts 64 x 64
Sentinel-2 image patches into ten land-use classes. Test accuracy 97.51%
on 4,050 unseen EuroSAT images.

## Run it on your computer

    pip install -r requirements.txt
    streamlit run app.py

## Files

- `app.py`: the Streamlit app
- `model.keras`: the trained network (Keras 3.13.2)
- `classes.json`, `metadata.json`: class names and test results
- `samples/`: one test image per class

## Data

Helber, P., Bischke, B., Dengel, A. and Borth, D. (2019) 'EuroSAT: a novel dataset and deep
learning benchmark for land use and land cover classification', IEEE Journal of Selected
Topics in Applied Earth Observations and Remote Sensing, 12(7), pp. 2217-2226.

## Limits

Trained only on European Sentinel-2 RGB patches at 10 m per pixel. Photos, drone images,
map screenshots or other resolutions will often be misread.
