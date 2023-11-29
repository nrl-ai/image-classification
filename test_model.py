import os, sys
import numpy as np
import cv2
from scipy.special import softmax


def main():
    image = cv2.imread("test/no_screw/2023-02-24_03-35-37_3.png")
    opencv_net = cv2.dnn.readNetFromONNX(
        "save_snapshot/resnet18/resnet18_best_model.onnx"
    )

    input_img = image.astype(np.float32)
    input_img = cv2.resize(input_img, (224, 224))
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]

    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(224, 224),
        mean=mean,
        swapRB=True,
        crop=True,
    )

    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    opencv_net.setInput(input_blob)
    out = opencv_net.forward()
    class_id = np.argmax(out)
    pred = softmax(out[0], axis=0)
    index = pred.argmax(axis=0)
    print(pred, index)


if __name__ == "__main__":
    main()
