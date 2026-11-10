from .generator import Generator, weights_init
from .discriminator import Discriminator
from .losses import GANLoss, compute_gradient_penalty


def train_gan(*args, **kwargs):
    """Import training lazily so module CLI execution has no eager import."""
    from .train_gan import train_gan as run
    return run(*args, **kwargs)

__all__ = ["Generator", "weights_init", "Discriminator", "GANLoss", "compute_gradient_penalty", "train_gan"]
