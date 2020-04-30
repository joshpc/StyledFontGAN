import torch
import torch.nn as nn
import torch.optim as optim

def build_font_shape_generator(glyph_size=(64, 64, 1), glyph_count=26, dimension=16):
  """
  Generator model for our GAN.

  Architecture is similar to DC-GAN with the exception of the input being an image.

  Inputs:
  - `image_size`: A triple (W, H, C) for the size of the images and number of channels. This model generates images the same size as the input (but for every character of the alphabet.)
  - `dimension`: Depth

  Output:
  -
  """

  return intermediate_generator_alt(glyph_size=glyph_size, glyph_count=glyph_count, dimension=dimension)

def simple_upscale_generator(dimension):
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

def intermediate_generator(glyph_size=(64, 64), glyph_count=26, dimension=16):
  linear_width = int(2 * dimension * glyph_size[0] / 4 * glyph_size[1] / 4)
  hidden_width = int(glyph_size[0] * glyph_size[1])
  # Final 2 * 2 is because we Conv Trans 3 times: 2x2 -> 4x4 -> 8x8 -> 16x16
  final_width = int(4 * dimension * glyph_count * 2 * 2)

  return nn.Sequential(
    # (1, 64, 64) -> (D, 32, 32)
    nn.Conv2d(1, dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    # (D, 32, 32) -> (2D, 16, 16)
    nn.Conv2d(dimension, 2 * dimension, 4, 2, 1),
    nn.LeakyReLU(0.2),

    # (2D, 16, 16) -> (1, 256 * D)
    Flatten(),

    # 256D -> 4096
    nn.Linear(in_features=linear_width, out_features=hidden_width),
    nn.ReLU(),

    # 4096 -> 16 * 16 * 26 * D
    # = 6656 D
    nn.Linear(hidden_width, final_width),
    nn.ReLU(),

    # 6656 D -> (4D, 16, 416)
    Unflatten(C=dimension * 4, H=2, W=2 * glyph_count),
    nn.BatchNorm2d(dimension * 4),

    # Fractionally Strided Conv 1
    nn.ConvTranspose2d(4 * dimension, 2 * dimension, 4, 2, 1), #4 * 4 * GC * 2D
    nn.BatchNorm2d(2 * dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 2
    nn.ConvTranspose2d(2 * dimension, dimension, 4, 2, 1), #8 * 8 * GC * D
    nn.BatchNorm2d(dimension),
    nn.ReLU(),

    # Fractionally Strided Conv 3
    # (D, 16, 416) -> (1, 16, 416)
    nn.ConvTranspose2d(dimension, 1, 4, 2, 1), # 16 * 16 * GC * 1
    nn.Sigmoid()
  )

def intermediate_generator_alt(glyph_size=(16, 16), glyph_count=26, dimension=512):
  conv_dimensions = [dimension, int(dimension / 2), int(dimension / 4)]
  fc_layer_widths = [
    int(conv_dimensions[2] * glyph_size[0] / 8 * glyph_size[1] / 8),
    int(glyph_size[0] * glyph_size[1]),
    int(glyph_size[0] / 8 * glyph_size[1] / 8 * glyph_count * dimension / 4)
  ]
  upconv_dimensions = [int(dimension / 4), int(dimension / 8), int(dimension / 16), 1]

  return nn.Sequential(
    # (1, 16, 16) -> (D, 8, 8)
    nn.Conv2d(1, conv_dimensions[0], 4, 2, 1),
    nn.LeakyReLU(0.2),

    # (D, 8, 8) -> (D/2, 4, 4)
    nn.Conv2d(conv_dimensions[0], conv_dimensions[1], 4, 2, 1),
    nn.LeakyReLU(0.2),

    # (D/2, 4, 4) -> (D/4, 2, 2)
    nn.Conv2d(conv_dimensions[1], conv_dimensions[2], 4, 2, 1),
    nn.LeakyReLU(0.2),

    # (D/4, 2, 2) -> (1, D)
    Flatten(),

    # D -> 256 (W * H)
    nn.Linear(in_features=fc_layer_widths[0], out_features=fc_layer_widths[1]),
    nn.ReLU(),

    # 256 -> 16 * 16 * 26 * D (W * H * GC * D/4)
    nn.Linear(fc_layer_widths[1], fc_layer_widths[2]),
    nn.ReLU(),

    # W * G * GC * D/8 -> (D/4, H, W * GC)
    Unflatten(C=upconv_dimensions[0], H=int(glyph_size[0] / 8), W=int(glyph_size[1] / 8) * glyph_count),
    nn.BatchNorm2d(upconv_dimensions[0]),

    # Fractionally Strided Conv 1
    nn.ConvTranspose2d(upconv_dimensions[0], upconv_dimensions[1], 4, 2, 1), #4 * 4 * GC * D/8
    nn.BatchNorm2d(upconv_dimensions[1]),
    nn.ReLU(),

    # Fractionally Strided Conv 2
    nn.ConvTranspose2d(upconv_dimensions[1], upconv_dimensions[2], 4, 2, 1), #8 * 8 * GC * D/16
    nn.BatchNorm2d(upconv_dimensions[2]),
    nn.ReLU(),

    # Fractionally Strided Conv 3
    # (D, 16, 416) -> (1, 16, 416)
    nn.ConvTranspose2d(upconv_dimensions[2], upconv_dimensions[3], 4, 2, 1), # 16 * 16 * GC * 1
    nn.Sigmoid()
  )

def build_font_shape_discriminator(image_size=(64, 1664), dimension=16):
  """
  PyTorch model implementing the GlyphGAN critic.

  Inputs:
  - `image_size`: The size of the entire alphabet (usually (H, W * 26))
  - `dimension`: The filter depth after each conv. Doubles per conv layer (1 - > 2 -> 4 -> 8)
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

def initialize_weights(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    nn.init.xavier_uniform_(m.weight.data)