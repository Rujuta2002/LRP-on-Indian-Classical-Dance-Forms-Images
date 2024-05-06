"""Per-image Layer-wise Relevance Propagation

This script uses a pre-trained VGG network from PyTorch's Model Zoo
to perform Layer-wise Relevance Propagation (LRP) on the images
stored in the 'input' folder.

NOTE: LRP supports arbitrary batch size. Plot function does currently support only batch_size=1.

"""
import argparse
import time
import pathlib

import torch
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt

from data import get_data_loader
from lrp import LRPModel

def plot_relevance_scores(
    x: torch.tensor, r: torch.tensor, name: str, config: argparse.Namespace
) -> None:
    """Plots results from layer-wise relevance propagation next to original image.

    Method currently accepts only a batch size of one.

    Args:
        x: Original image.
        r: Relevance scores for original image.
        name: Image name.
        config: Argparse namespace object.

    """
    output_dir = config.output_dir

    max_fig_size = 20

    _, _, img_height, img_width = x.shape
    max_dim = max(img_height, img_width)
    fig_height, fig_width = (
        max_fig_size * img_height / max_dim,
        max_fig_size * img_width / max_dim,
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height))

    x = x[0].squeeze().permute(1, 2, 0).detach().cpu()
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    axes[0].imshow(x)
    axes[0].set_axis_off()

    r_min = r.min()
    r_max = r.max()
    r = (r - r_min) / (r_max - r_min)
    axes[1].imshow(r, cmap="afmhot")
    axes[1].set_axis_off()

    fig.tight_layout()
    plt.savefig(f"{output_dir}/image_{name}.png", bbox_inches="tight")
    plt.close(fig)

def per_image_lrp(config: argparse.Namespace) -> None:
    """Test function that plots heatmaps for images placed in the input folder.

    Images have to be placed in their corresponding class folders.

    Args:
        config: Argparse namespace object.

    """
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    data_loader = get_data_loader(config)

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)

    lrp_model = LRPModel(model=model, top_k=config.top_k)

    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        # y = y.to(device)  # here not used as method is unsupervised.

        t0 = time.time()
        r = lrp_model.forward(x)
        print("{time:.2f} FPS".format(time=(1.0 / (time.time() - t0))))

        plot_relevance_scores(x=x, r=r, name=str(i), config=config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Input directory.",
        default="./input/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory.",
        default="./output/",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", help="Batch size.", default=1, type=int
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="Device.",
        choices=["gpu", "cpu"],
        default="gpu",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--top-k",
        dest="top_k",
        help="Proportion of relevance scores that are allowed to pass.",
        default=0.02,
        type=float,
    )
    parser.add_argument(
        "-r",
        "--resize",
        dest="resize",
        help="Resize image before processing.",
        default=0,
        type=int,
    )

    config = parser.parse_args()

    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    per_image_lrp(config=config)