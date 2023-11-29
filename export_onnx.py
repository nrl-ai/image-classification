import sys, os
import argparse
import numpy as np
import torch
from torch.nn import Linear, Sequential
from torchvision import models
from pathlib import Path
from default_config import get_export_config


class Convert_model(object):
    def __init__(self, conf):
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_infer = self.conf.net
        self.input_size = self.conf.input_size
        self.model = self._load_model()

    def _load_model(self):
        if self.net_infer == "resnet50":
            model = models.resnet50(pretrained=False)
            model.fc = Sequential(
                model.fc,
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.net_infer == "resnet34":
            model = models.resnet34(pretrained=False)
            model.fc = Sequential(
                model.fc,
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.net_infer == "resnet18":
            model = models.resnet18(pretrained=False)
            model.fc = Sequential(
                model.fc,
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.net_infer == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=False)
            model.classifier.add_module(
                str(len(model.classifier)),
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.net_infer == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=False)
            model.classifier.add_module(
                str(len(model.classifier)),
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        else:
            model = models.mobilenet_v3_large(pretrained=False)
            model.classifier.add_module(
                str(len(model.classifier)),
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )

        model.load_state_dict(torch.load(self.conf.path_model))
        model.to(self.device)
        model.eval()
        print("[INFO] Loading Classification Model ....")
        return model

    def export_model(self):
        batch_size = 1
        x = torch.randn(
            batch_size, 3, self.input_size, self.input_size, requires_grad=True
        )
        print("[INFO] Exporting ONNX Model ....")
        torch.onnx.export(
            self.model,
            x.to(self.device),
            self.conf.path_model.replace(".pth", ".onnx"),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        print("[INFO] Exporting ONNX Finish !")


def parse_args():
    """parsing and configuration"""
    desc = "Export a PyTorch model to ONNX"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--network",
        type=str,
        help="[resnet18, resnet34, resnet50, mobilenet_v2, mobilenet_v3_small]",
    )
    parser.add_argument("--num_classes", type=int, help="The number of classes")
    parser.add_argument("--input_size", type=int, help="input size")
    parser.add_argument("--path_model", type=str, help="Path weight")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    conf = get_export_config(args)

    convert_onnx = Convert_model(conf)
    convert_onnx.export_model()
