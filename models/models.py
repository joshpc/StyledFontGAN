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

  #TODO: Okay this actually doesn't quite make sense.
  # We have high res images (64x64) that we downsample to 16x16 greyscale only to upscale immediately after?
  # We need to take advantage of the large image size to retain as much original information as possible.
  # We're also going to be generating a large image from a small one.
  #
  # 64 x 64 -----> 64 x 1664
  # Determine what steps we need to take to from A to B
  # Open questions:
  # - Is it better to downsample, do a ton of work, then scale up and sharpen?
  # - Maybe shrink to an internal representation, then backwards conv?

  # Attempt 1
  # Simply upscale. Go big. Go from 64x64 to 64x1664
  #
  # Attempt 2
  # Conv downwards. Build an internal representation (linear layers.) Then work backwards.
  #
  # Attempt 3
  # Go small, end small. Use a separate network to grow.

  return simple_upscale_degenrator(dimension)

def simple_upscale_degenrator(dimension):
  """
  A generator that performs several ConvTranpsose2D Operations to upscale an image from `individual_image_size` to `final_image_size`. The dimensions of `final_image_size` must be an integer multiple of `individual_image_size.`

  Inputs:
  - `individual_image_size`: (W, H) the size of the images provided (and the expected output size.)
  - `dimension`: This imapcts the scale of the number of features in the upscale.

  Output:
  - An image that is 256 * 512
  """

  return nn.Sequential(
    # We start with a simple image of size (W, H)
    # K = 3, S = 1, P = 1 -- Get a full view of the image.
    # This changes an image from 64x64 to 128 * 128 (2x2 grid = 4 images) - We start in a high dimension
    nn.ConvTranspose2d(in_channels=1, out_channels=(8 * dimension), kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    nn.ReLU(),

    # This changes an image from 128 * 128 to 256 * 256 (4x4 grid = 16 images) - Scale down the dimensionality
    nn.ConvTranspose2d(in_channels=(8 * dimension), out_channels=(4 * dimension), kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    nn.ReLU(),

    # # Reduce dimensionality without changing the image. Stays at 4x4 grid.
    nn.ConvTranspose2d(in_channels=(4 * dimension), out_channels=(2 * dimension), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(),

    # # This changes an image from 256 * 256 to 256 * 512 (4x8 grid = 32 images)
    nn.ConvTranspose2d(in_channels=(2 * dimension), out_channels=dimension, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)),
    nn.ReLU(),

    # # Reduce dimensionality back to 1!
    nn.ConvTranspose2d(in_channels=dimension, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.Sigmoid()
  )

def build_font_shape_discriminator(image_size=(64, 1664), dimension=16):
  """
  PyTorch model implementing the GlyphGAN critic.

  Inputs:
  - `image_size`: The size of the images (W, H) tuple. (32, 32)
  - `dimension`: The dpeth of the noise, defaults to 16.
  """

  output_size = int(8 * dimension * (image_size[0] / 16) * (image_size[1] / 16))

  return nn.Sequential(
    nn.Conv2d(1, dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(dimension, 2 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(2 * dimension, 4 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(4 * dimension, 8 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    Flatten(),

    nn.Linear(output_size, 1), # Default size will be 3328 variable linear layer
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