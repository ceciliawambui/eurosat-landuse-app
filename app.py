"""EuroSAT land-use classifier. Run locally with: streamlit run app.py"""
import json
from pathlib import Path

import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

HERE = Path(__file__).parent
CLASSES = json.loads((HERE / "classes.json").read_text())
INFO = json.loads((HERE / "metadata.json").read_text())
NOT_SURE_BELOW = INFO["not_sure_below"]


@st.cache_resource  # load the model once, not on every click
def load_model():
    return keras.models.load_model(HERE / "model.keras", compile=False)


def to_patch(image):
    """Any picture -> the 64 x 64 RGB array the model was trained on."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.size != (64, 64):
        image = ImageOps.fit(image, (64, 64), method=Image.BILINEAR)  # centre square, then shrink
    return np.asarray(image, dtype="float32")


def grad_cam(model, patch, class_index):
    """Grad-CAM exactly as in Stage 9: cut the model after 'last_relu'."""
    names = [layer.name for layer in model.layers]
    cut = names.index("last_relu")
    fmap = tf.convert_to_tensor(patch[None])
    for layer in model.layers[:cut + 1]:
        if not isinstance(layer, keras.layers.InputLayer):
            fmap = layer(fmap, training=False)
    dense = model.layers[-1]
    with tf.GradientTape() as tape:
        tape.watch(fmap)
        h = fmap
        for layer in model.layers[cut + 1:-1]:
            h = layer(h, training=False)
        score = (tf.matmul(h, dense.kernel) + dense.bias)[0, class_index]
    weights = tf.reduce_mean(tape.gradient(score, fmap), axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], (64, 64))[..., 0].numpy()


def heat_overlay(patch, cam):
    """Same colours as the Stage 9 figures: red = strong evidence, blue = none."""
    c = cam[..., None]
    jet = np.concatenate([np.clip(1.5 - np.abs(4 * c - 3), 0, 1),
                          np.clip(1.5 - np.abs(4 * c - 2), 0, 1),
                          np.clip(1.5 - np.abs(4 * c - 1), 0, 1)], axis=-1) * 255
    return 0.55 * patch + 0.45 * jet


def enlarge(array, size=256):
    return Image.fromarray(np.clip(array, 0, 255).astype("uint8")).resize((size, size), Image.NEAREST)


st.set_page_config(page_title="EuroSAT land-use classifier")
st.title("Land-use classifier for satellite image patches")
st.write("Pick a test image or upload a 64 x 64 Sentinel-2 patch. The model sorts it into "
         "one of the ten land-use classes of the EuroSAT dataset.")

with st.sidebar:
    st.header("About the model")
    st.write(INFO["model"])
    st.write(f"Test accuracy {INFO['test_accuracy']:.2%} (macro-F1 {INFO['test_macro_f1']:.3f}) "
             f"on {INFO['n_test']:,} images it never saw in training.")
    st.write("Trained on EuroSAT RGB (Helber et al., 2019): 64 x 64 pixel Sentinel-2 "
             "patches of Europe, 10 m per pixel. It has never seen phone photos or "
             "zoomed-in map screenshots, so it may get those wrong.")

source = st.radio("Image", ["Test image", "Upload your own"], horizontal=True)
show_cam = st.toggle("Also show where the model found its evidence (Grad-CAM)")
if source == "Upload your own":
    upload = st.file_uploader("JPG or PNG", type=["jpg", "jpeg", "png"])
    image = Image.open(upload) if upload is not None else None
else:
    samples = sorted((HERE / "samples").glob("*.png"))
    choice = st.selectbox("Test image (its true class)", samples, format_func=lambda p: p.stem)
    image = Image.open(choice)

if image is None:
    st.stop()

resized = image.size != (64, 64)
patch = to_patch(image)
model = load_model()
probs = model.predict(patch[None], verbose=0)[0]
order = np.argsort(probs)[::-1]
best = int(order[0])

left, right = st.columns(2)
with left:
    st.image(enlarge(patch), width=256, caption="What the model sees (64 x 64, enlarged)")
    if resized:
        st.caption("Your image was cropped to a square and shrunk to 64 x 64.")
    if show_cam:
        cam = grad_cam(model, patch, best)
        st.image(enlarge(heat_overlay(patch, cam)), width=256,
                 caption=f"Evidence for {CLASSES[best]}: red strong, blue none")
with right:
    if probs[best] >= NOT_SURE_BELOW:
        st.subheader(CLASSES[best])
    else:
        st.subheader("Not sure")
        st.write(f"Best guess: {CLASSES[best]}. On the test set the model was right only "
                 f"{INFO['accuracy_below']:.0%} of the time when it was this unsure.")
    for k in order[:3]:
        st.progress(float(probs[k]), text=f"{CLASSES[k]}: {probs[k]:.1%}")
