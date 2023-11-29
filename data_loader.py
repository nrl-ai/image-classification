import os
import threading
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Any, Callable, Optional, Tuple


class CustomImage(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__(root, transform=transform)
        self.load_all_images(nb_threads=64)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, target = self.images[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def mono_load(self, ids: list):
        while len(ids) > 0:
            index = ids.pop(0)
            path, target = self.samples[index]
            sample = self.loader(path)
            self.images.append((sample, target))
            percent = round(100 * len(ids) / len(self.samples))
            print("Loading: {} %".format(100 - percent), end="\r")

    def load_all_images(self, nb_threads=16):
        self.images = []
        threads = []
        ids = list(range(len(self.samples)))
        for _ in range(nb_threads):
            t = threading.Thread(target=self.mono_load, args=(ids,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()


def get_data_loaders(conf):
    train_transform = transforms.Compose(
        [
            transforms.Resize((conf.input_size, conf.input_size)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomRotation(90),
            transforms.RandomAffine(0),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((conf.input_size, conf.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_images = CustomImage(os.path.join(conf.data_dir, "train"), train_transform)
    val_images = CustomImage(os.path.join(conf.data_dir, "val"), val_transform)
    test_images = CustomImage(os.path.join(conf.data_dir, "test"), val_transform)
    train_loader = DataLoader(
        train_images,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.nb_workers,
    )
    val_loader = DataLoader(
        val_images,
        batch_size=int(conf.batch_size / 4),
        shuffle=False,
        num_workers=conf.nb_workers,
    )
    test_loader = DataLoader(
        test_images,
        batch_size=int(conf.batch_size / 4),
        shuffle=False,
        num_workers=conf.nb_workers,
    )
    data_sizes = {
        "train": len(train_images),
        "val": len(val_images),
        "test": len(test_images),
    }
    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    labels = sorted(os.listdir(os.path.join(conf.data_dir, "train")))

    classids_labels = {i: labels[i] for i in range(len(labels))}
    return data_loaders, data_sizes, classids_labels
