import io
from typing import ByteString, Callable
import numpy as np
import numpy.typing as npt
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance
from tensorflow.keras.applications.mobilenet import (
    decode_predictions as mobilenet_decode_predictions,
)
from tensorflow.keras.applications.mobilenet import (
    preprocess_input as mobilenet_preprocess_input,
)
from tensorflow.keras.applications.vgg16 import (
    decode_predictions as vgg16_decode_predictions,
)
from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg16_preprocess_input,
)
from tensorflow.keras.preprocessing import image

IMAGENET_INPUT_SIZE = (224, 224)
IMAGENET_INPUT_SHAPE = [224, 224, 3]


def bytes_to_array(image_bytes: ByteString) -> npt.ArrayLike:
    """Converts image stored in bytes into a Numpy array
    
    Args:
        image_bytes (ByteString): Image stored as bytes
    
    Returns:
        npt.ArrayLike: Image stored as Numpy array
    """
    return np.array(Image.open(io.BytesIO(image_bytes)))


@st.cache_resource
def load_vgg16() -> tf.keras.Model:
    """Loads pre-trained VGG16 Keras model
    
    Returns:
        tf.keras.Model: VGG-16 model
    """
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling="avg",  # Use average pooling for better feature extraction
        classes=1000,
    )
    
    return model


@st.cache_resource
def load_mobilenet() -> tf.keras.Model:
    """Loads pre-trained MobileNet Keras model
    
    Returns:
        tf.keras.Model: MobileNet model
    """
    model = tf.keras.applications.MobileNet(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling="avg",  # Use average pooling for better feature extraction
        classes=1000,
    )
    
    return model


SUPPORTED_MODELS = {
    "VGG-16": {
        "load_model": load_vgg16,
        "preprocess_input": vgg16_preprocess_input,
        "decode_predictions": vgg16_decode_predictions,
    },
    "MobileNet": {
        "load_model": load_mobilenet,
        "preprocess_input": mobilenet_preprocess_input,
        "decode_predictions": mobilenet_decode_predictions,
    },
}


def enhance_image(img_array: npt.ArrayLike) -> npt.ArrayLike:
    """Enhance image quality to improve model performance
    
    Args:
        img_array (npt.ArrayLike): Input image array
        
    Returns:
        npt.ArrayLike: Enhanced image array
    """
    # Convert to PIL for enhancement
    img = Image.fromarray(img_array)
    
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    return np.array(img)


@st.cache_data
def prepare_image(img_array: npt.ArrayLike, _model_preprocess: Callable) -> npt.ArrayLike:
    """Prepare any image so that it can be fed into a model predict() function.
    This includes:
    - converting to RGB channels
    - enhancing image quality
    - resizing to the appropriate image size expected by the model
    - reshaping to have the proper ordering of dimensions
    - preprocess the image according to the model's weights
    
    Args:
        img_array (npt.ArrayLike): Input image
        _model_preprocess (Callable): Model preprocessing function
    
    Returns:
        npt.ArrayLike: Image ready to be fed into predict()
    """
    # Enhance image quality
    img_array = enhance_image(img_array)
    
    img = Image.fromarray(img_array)
    img = img.convert("RGB")
    
    # Use BICUBIC interpolation instead of NEAREST for better quality
    img = img.resize(IMAGENET_INPUT_SIZE, Image.BICUBIC)
    
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = _model_preprocess(img)
    img = img.reshape(*([1] + IMAGENET_INPUT_SHAPE))
    return img


def predict_with_confidence(model: tf.keras.Model, prepared_img: npt.ArrayLike, 
                           decode_fn: Callable, top_k: int = 5) -> list:
    """Make a prediction with the model and add some confidence boosting
    
    Args:
        model: Model to use for prediction
        prepared_img: Preprocessed image
        decode_fn: Function to decode predictions
        top_k: Number of top predictions to return
        
    Returns:
        List of decoded predictions
    """
    # Multiple prediction passes with small variations in image
    preds = []
    
    # Original prediction
    preds.append(model.predict(prepared_img))
    
    # Slightly different versions with minor crops
    # This simulates test-time augmentation without major code changes
    img = prepared_img.copy()
    # Crop 1: crop 5% from right and bottom
    crop1 = img[:, :218, :218, :]
    crop1 = tf.image.resize(crop1, IMAGENET_INPUT_SIZE).numpy()
    preds.append(model.predict(crop1))
    
    # Crop 2: crop 5% from left and top
    crop2 = img[:, 6:, 6:, :]
    crop2 = tf.image.resize(crop2, IMAGENET_INPUT_SIZE).numpy()
    preds.append(model.predict(crop2))
    
    # Average predictions for more robust results
    avg_preds = np.mean(preds, axis=0)
    
    # Return decoded predictions
    return decode_fn(avg_preds, top=top_k)[0]