"""Run the EcoSort ONNX model on one image.

Example:
    python predict.py --image path/to/image.png
"""

import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image


CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((96, 96), resample=Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = (array - MEAN) / STD
    return np.expand_dims(array, axis=0).astype(np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one recyclable-waste image")
    parser.add_argument("--image", required=True, help="Path to a PNG/JPG image")
    parser.add_argument("--model", default="model.onnx", help="Path to model.onnx")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: preprocess(args.image)})[0][0]
    probabilities = softmax(logits)
    prediction = int(np.argmax(probabilities))

    print(f"Prediction: {CLASSES[prediction]}")
    print("Class probabilities:")
    for name, probability in sorted(zip(CLASSES, probabilities), key=lambda item: item[1], reverse=True):
        print(f"  {name:9s}: {probability * 100:6.2f}%")


if __name__ == "__main__":
    main()
