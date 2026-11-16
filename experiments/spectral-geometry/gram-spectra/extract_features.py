"""Extract all pooled CIFAR-10 vectors with the official ImageNet ResNet-50."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import pickle
import tarfile

import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet50_Weights, resnet50


def extract(cache):
    archive = (
        cache / "mirror-cifar.tar.gz"
        if (cache / "mirror-cifar.tar.gz").exists()
        else cache / "cifar-10-python.tar.gz"
    )
    checkpoint = cache / "resnet50-0676ba61.pth"
    for path, algorithm, expected in [
        (archive, "md5", "c58f30108f718f92721af3b95e74349a"),
        (checkpoint, "sha256", "0676ba61"),
    ]:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, algorithm).hexdigest()
        if not digest.startswith(expected):
            raise ValueError(f"Checksum mismatch for {path.name}: {digest}")
        print(path.name, digest, flush=True)

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50(weights=None)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    assert net.fc.in_features == 2048
    net.fc = torch.nn.Identity()
    net.eval().to(device=device, memory_format=torch.channels_last)
    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

    def preprocess(image):
        return transform(Image.fromarray(image))

    with (
        tarfile.open(archive, "r:gz") as data,
        ThreadPoolExecutor(4) as pool,
        torch.inference_mode(),
    ):
        for name in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            output = cache / f"resnet50_{name}_features.npy"
            if output.exists():
                assert np.load(output, mmap_mode="r").shape == (10000, 2048)
                continue
            with data.extractfile(f"cifar-10-batches-py/{name}") as batch_file:
                batch = pickle.load(batch_file, encoding="bytes")
            images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            features = np.empty((len(images), 2048), dtype=np.float32)
            for start in range(0, len(images), 96):
                tensors = list(pool.map(preprocess, images[start : start + 96]))
                inputs = torch.stack(tensors).to(
                    device=device, memory_format=torch.channels_last
                )
                features[start : start + len(tensors)] = net(inputs).cpu().numpy()
            assert features.shape == (10000, 2048) and np.isfinite(features).all()
            temporary = output.with_suffix(".tmp.npy")
            np.save(temporary, features)
            os.replace(temporary, output)
            print("Extracted", name, features.shape, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    extract(parser.parse_args().cache)
