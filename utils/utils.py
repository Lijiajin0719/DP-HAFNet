import os

import numpy as np
from matplotlib import pyplot as plt
from torch import optim


def create_optimizer(model, args):
    """Create optimizer"""
    return optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-08,
        weight_decay=args.weight_decay
    )


def create_scheduler(optimizer, args):
    """Create cosine cyclic learning rate scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.lr_period,
        eta_min=args.lr_min
    )


def save_visualization(data):
    """Save visualization image"""
    img_log = 20 * np.log10(np.abs(data) + 0.001)
    img_log = img_log - np.amax(img_log)
    return img_log


def save_concatenated_image(images, filename, save_dir, titles=None):
    """Save concatenated image"""
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))

    if len(images) == 1:
        axes = [axes]

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches='tight')
    plt.close()