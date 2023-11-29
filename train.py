import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pathlib
import shutil
import os
import argparse
import torch
import json
from tqdm import tqdm
from torch import optim
from torch.nn import CrossEntropyLoss, Linear, Sequential
from data_loader import get_data_loaders
import torchvision.models as models
import matplotlib.pyplot as plt
from default_config import get_default_config


class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.board_load_every = conf.board_load_every
        self.save_every = conf.save_every
        self.steps = {"train": 0, "val": 0}
        self.data_loaders, self.data_sizes, self.classids_labels = get_data_loaders(
            conf
        )
        self.model_path = conf.model_path
        self.metrics_path = os.path.join(self.model_path, "metrics.json")

    def train(self):
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.criterion = CrossEntropyLoss()
        self.model = self._define_network()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.conf.lr, weight_decay=5e-4
        )
        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, -1
        )

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        counter = []
        train_acc_hist = []
        val_acc_hist = []
        test_acc_hist = []
        best_acc = 0.0
        best_acc_test = 0.0
        for e in range(self.conf.epochs):
            counter.append(e)
            print("epoch {} started".format(e + 1))
            print("lr: ", self.schedule_lr.get_lr())

            self.model.train()
            training_loss = 0.0
            training_corrects = 0
            for images, labels in iter(self.data_loaders["train"]):
                images = images.cuda()
                labels = labels.cuda()
                self.optimizer.zero_grad()
                # forward
                outputs = self.model.forward(images)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                training_loss += loss.item() * images.size(0)
                training_corrects += torch.sum(preds == labels.data)
                # backward
                loss.backward()
                self.optimizer.step()
                # tensorboard
                self.steps["train"] += 1
                print(
                    "Epoch: {}/{}... ".format(e + 1, self.conf.epochs),
                    "Loss: {:.4f}".format(loss.item()),
                )
            training_acc = float(training_corrects.double() / self.data_sizes["train"])
            train_acc_hist.append(training_acc)

            self.schedule_lr.step()

            # validation phase
            self.model.eval()
            with torch.no_grad():
                validation_loss = 0.0
                validation_corrects = 0
                for images, labels in iter(self.data_loaders["val"]):
                    self.steps["val"] += 1
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = self.model.forward(images)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    validation_loss += loss.item() * images.size(0)
                    validation_corrects += torch.sum(preds == labels.data)
                validation_acc = float(
                    validation_corrects.double() / self.data_sizes["val"]
                )
                val_acc_hist.append(validation_acc)

                test_corrects = 0
                for images, labels in iter(self.data_loaders["test"]):
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = self.model.forward(images)
                    _, preds = torch.max(outputs, 1)
                    test_corrects += torch.sum(preds == labels.data)
                test_acc = float(test_corrects.double() / self.data_sizes["test"])
                test_acc_hist.append(test_acc)

                if validation_acc >= best_acc and test_acc >= best_acc_test:
                    best_model_path = os.path.join(
                        self.conf.model_path, "model.pth"
                    )
                    torch.save(self.model.state_dict(), best_model_path)
                    best_acc = validation_acc
                    best_acc_test = test_acc

                print(
                    "Epoch: {}/{}... ".format(e + 1, self.conf.epochs),
                    "Validation Loss: {:.4f}".format(validation_loss),
                    "Validation Acc: {:.4f}".format(validation_acc),
                    "Test Acc: {:.4f}".format(test_acc),
                )

            plt.figure(figsize=(9.6, 6.4))
            plt.plot(counter, train_acc_hist, label="Training", linestyle="-")
            plt.plot(counter, val_acc_hist, label="Validation", linestyle="--")
            plt.plot(counter, test_acc_hist, label="Test", linestyle="-.")
            plt.grid(True)
            plt.ylim([0, 1.1])
            plt.legend(loc="lower right")
            plt.savefig(
                os.path.join(self.conf.model_path, f"graph.png")
            )
            plt.close()

            with open(self.metrics_path, "w") as fp:
                json.dump(
                    {
                        "train_acc": train_acc_hist,
                        "val_acc": val_acc_hist,
                        "test_acc": test_acc_hist,
                    },
                    fp,
                )

        with open(os.path.join(self.conf.model_path, "class_names.json"), "w") as fp:
            json.dump(self.classids_labels, fp)

    def _define_network(self):
        if self.conf.net == "resnet50":
            model = models.resnet50(pretrained=False)
            if conf.path_pretrain != "None":
                checkpoint = torch.load(conf.path_pretrain)
            else:
                checkpoint = torch.load("pretrains/resnet50-19c8e357.pth")
            model.load_state_dict(checkpoint, strict=False)
            model.fc = Sequential(
                model.fc,
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.conf.net == "resnet34":
            model = models.resnet34(pretrained=False)
            if conf.path_pretrain != "None":
                checkpoint = torch.load(conf.path_pretrain)
            else:
                checkpoint = torch.load("pretrains/resnet34-333f7ec4.pth")
            model.load_state_dict(checkpoint, strict=False)
            model.fc = Sequential(
                model.fc,
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.conf.net == "resnet18":
            model = models.resnet18(pretrained=False)
            if conf.path_pretrain != "None":
                checkpoint = torch.load(conf.path_pretrain)
            else:
                checkpoint = torch.load("pretrains/resnet18-5c106cde.pth")
            model.load_state_dict(checkpoint, strict=False)
            model.fc = Sequential(
                model.fc,
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.conf.net == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=False)
            if conf.path_pretrain != "None":
                checkpoint = torch.load(conf.path_pretrain)
            else:
                checkpoint = torch.load("pretrains/mobilenet_v2-b0353104.pth")
            model.load_state_dict(checkpoint, strict=False)
            model.classifier.add_module(
                str(len(model.classifier)),
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        elif self.conf.net == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=False)
            if conf.path_pretrain != "None":
                checkpoint = torch.load(conf.path_pretrain)
            else:
                checkpoint = torch.load("pretrains/mobilenet_v3_small-047dcff4.pth")
            model.load_state_dict(checkpoint, strict=False)
            model.classifier.add_module(
                str(len(model.classifier)),
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )
        else:
            model = models.mobilenet_v3_large(pretrained=False)
            if conf.path_pretrain != "None":
                checkpoint = torch.load(conf.path_pretrain)
            else:
                checkpoint = torch.load("pretrains/mobilenet_v3_large-8738ca79.pth")
            model.load_state_dict(checkpoint, strict=False)
            model.classifier.add_module(
                str(len(model.classifier)),
                Linear(in_features=1000, out_features=self.conf.nb_classes, bias=True),
            )

        if self.conf.net in ["resnet18", "resnet34", "resnet50"]:
            torch.set_grad_enabled(True)
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.features.parameters():
                param.requires_grad = False

        return model.cuda()


def parse_args():
    """parsing and configuration"""
    desc = "Object Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--network",
        type=str,
        default="resnet18",
        help="[resnet18, resnet34, resnet50, mobilenet_v2, mobilenet_v3_small]",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="The number of classes"
    )
    parser.add_argument("--input_size", type=int, default=224, help="input size")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="The learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--dataset", type=str, default="dataset/mvtec", help="Path dataset"
    )
    parser.add_argument(
        "--path_pretrain",
        type=str,
        default="None",
        help="Path weight pretrain [None, path_weights]",
    )
    parser.add_argument(
        "--job_name", type=str, default="resnet18", help="The name of experiment"
    )
    parser.add_argument(
        "--model_path", type=str, default="out_snapshot", help="Path model"
    )
    # parser.add_argument('-d', "--device_ids", type=str, default="1", help="Which gpu id: 0,1,2,3")
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    return args


if __name__ == "__main__":
    args = parse_args()
    conf = get_default_config(args)
    trainer = Trainer(conf)
    trainer.train()
    print("Done!")
