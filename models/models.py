import torch
import torch.nn as nn
import torch.optim as optim

def build_font_shape_generator(image_size=(64, 64, 1), dimension=16):
  #TODO: Describe how flatten/unflattening should work.
  """
  Generator model for our GAN.

  Architecture is similar to DC-GAN with the exception of the input being an image.

  Inputs:
  - `image_size`: A triple (W, H, C) for the size of the images and number of channels. This model generates images the same size as the input (but for every character of the alphabet.)
  - `dimension`: Depth

  Output:
  -
  """

  return nn.Sequential(
    # # 128 features
    # nn.BatchNorm2d(8 * dimension),

    nn.ConvTranspose2d(in_channels=image_size[2], out_channels=(8 * dimension), kernel_size=5),

    # Fractionally Strided Conv 1
    # 128 features -> 64 features
    nn.ConvTranspose2d(8 * dimension, 4 * dimension, 4, 2, 1),
    nn.BatchNorm2d(4 * dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 2
    # 64 -> 32 features
    nn.ConvTranspose2d(4 * dimension, 2 * dimension, 4, 2, 1),
    nn.BatchNorm2d(2 * dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 3
    # 32 -> 16 features
    nn.ConvTranspose2d(2 * dimension, dimension, 4, 2, 1),
    nn.BatchNorm2d(dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 4
    # 16 features
    nn.ConvTranspose2d(dimension, image_size[2], 4, 2, 1),
    nn.Sigmoid()
  )

def build_font_shape_discriminator(image_size=(32, 32), dimension=16):
  """
  PyTorch model implementing the GlyphGAN critic.

  Inputs:
  - `image_size`: The size of the images (W, H) tuple. (32, 32)
  - `dimension`: The dpeth of the noise, defaults to 16.
  """

  output_size = int(8 * dimension * (image_size[0] / 16) * (image_size[1] / 16))

  return nn.Sequential(
    nn.Conv2d(image_size[2], dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(dimension, 2 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(2 * dimension, 4 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(4 * dimension, 8 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    Flatten(),

    nn.Linear(output_size, 1),
    nn.Sigmoid()
  )

def get_optimizer(model, learning_rate=2e-4, beta1=0.5, beta2=0.99):
    """
    Adam optimizer for model

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
    return optimizer

class Flatten(nn.Module):
  def forward(self, x):
    N, _, _, _ = x.size() # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
  """
  An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
  to produce an output of shape (N, C, H, W).
  """
  def __init__(self, N=-1, C=128, H=7, W=7):
      super(Unflatten, self).__init__()
      self.N = N
      self.C = C
      self.H = H
      self.W = W

  def forward(self, x):
      return x.view(self.N, self.C, self.H, self.W)