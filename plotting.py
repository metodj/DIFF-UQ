import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def display_images(
    image_dir, idx, idx_reverse, num_images=25, figsize=(10, 10), plot_type="best", save=None, low_res=False, dpi=600
):
    """
    Display a grid of images from the specified directory

    Parameters:
    image_dir (str): Directory containing PNG images
    num_images (int): Number of images to display (should be a perfect square)
    figsize (tuple): Figure size in inches
    """
    # Get list of PNG files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    image_files = sorted(image_files)

    idx = np.load(idx)
    idx = np.argsort(idx)
    if idx_reverse:
        idx = idx[::-1]
    image_files = [image_files[idx[i]] for i in range(len(idx))]

    # Calculate grid dimensions
    grid_size = int(np.sqrt(num_images))

    # Create subplot grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize, gridspec_kw={"wspace": 0, "hspace": 0})

    # Flatten axes for easier iteration
    axes = axes.ravel()

    if plot_type == "best":
        print(idx[:num_images])
        pass
    elif plot_type == "worst":
        print(idx[-num_images:])
        image_files = image_files[::-1]
    else:
        raise ValueError()

    # Display images
    for idx, (ax, img_file) in enumerate(zip(axes, image_files[:num_images])):
        # Read image
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)

        # Display image
        ax.imshow(img)
        ax.axis("off")
        # ax.set_title(f'Image {idx + 1}')

    plt.subplots_adjust(wspace=0, hspace=0)
    if save:
        if low_res:
            plt.savefig(save, bbox_inches="tight", pad_inches=0, dpi=dpi)
        else:
            plt.savefig(save, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_best_worst(exp_img, imgs, exp, unc, dpi):
    for plot_type in ["best", "worst"]:
        paths = display_images(exp_img + imgs, exp + unc, False, plot_type=plot_type, low_res=True, dpi=dpi)
