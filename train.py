import time
import random
from operator import itemgetter

import torch
import collections

# Visualization
from livelossplot import PlotLosses
from util import show_grayscale_image

def train(D, G, D_optimizer, G_optimizer, D_loss, G_loss, data_loader, options):
  """
  Inputs:
    - `options`: A dictionary of options to configure the GAN. Required values:
                    `batch_size` - (int) The size of each batch.
                    `epoch_count` - (int) The number of epochs to run.
                    `data_type` -
                    `glyph_size` - (tuple or triple, [int, int, (int)]) The size of the image (H, W, C)
                    `glyphs_per_image` - (int) The number of glyphs found on each image

  """
  epoch_count = options['epoch_count']
  losses = collections.defaultdict(list)
  loss_plot = PlotLosses()

  real_test, static_test = prepare_static_test(data_loader, options)
  show_grayscale_image(real_test[0].cpu())
  show_grayscale_image(static_test[0].cpu())
  show_grayscale_image(G(static_test)[0].cpu())

  for _ in range(epoch_count):
    train_epoch(D, G, D_optimizer, G_optimizer, D_loss, G_loss, data_loader, losses, options)
    record_losses(loss_plot, losses)

    show_grayscale_image(real_test[0].cpu())
    show_grayscale_image(static_test[0].cpu())
    show_grayscale_image(G(static_test)[0].cpu())

def train_epoch(D, G, D_optimizer, G_optimizer, D_loss, G_loss, data_loader, losses, options):
  steps = 0
  batch_size = options['batch_size']
  data_type = options['data_type']

  for data in data_loader:
    if len(data) % batch_size != 0:
      continue
    data = data.type(data_type)

    steps += 1

    train_discriminator(D, G, D_optimizer, D_loss, data, losses, options)

    # TODO: Parameterize
    if steps % 5 == 0:
      train_generator(D, G, G_optimizer, G_loss, data, losses, options)

def train_generator(D, G, G_optimizer, G_loss, data, losses, options):
  """
  Executes one interation of training for the generator. This is a classic GAN setup.

  No return value.
  """
  glyph_size, glyphs_per_image = itemgetter('glyph_size', 'glyphs_per_image')(options)

  G_optimizer.zero_grad()

  # Prepare our data. We only use the letter A to seed this entire process.
  generator_input = prepare_generator_input(data, glyph_size, glyphs_per_image)
  generated_data = reshape_generated_data(G(generator_input), glyph_size, glyphs_per_image)
  real_data = reshape_real_data(data, glyph_size, glyphs_per_image)

  loss = G_loss(D, G, real_data, generated_data, losses, options)
  loss.backward()
  losses['G'].append(loss.data)

  G_optimizer.step()

def train_discriminator(D, G, D_optimizer, D_loss, data, losses, options):
  """
  Executes one iteration of training for the discriminator.

  No return value.
  """
  glyph_size, glyphs_per_image = itemgetter('glyph_size', 'glyphs_per_image')(options)

  D_optimizer.zero_grad()

  # Prepare the data
  generator_input = prepare_generator_input(data, glyph_size, glyphs_per_image)
  generated_data = reshape_generated_data(G(generator_input), glyph_size, glyphs_per_image)
  real_data = reshape_real_data(data, glyph_size, glyphs_per_image)

  # Calculate the loss
  loss = D_loss(D, real_data, generated_data, losses, options)
  loss.backward()
  losses['D'].append(loss.data)

  # Perform backwards pass
  D_optimizer.step()

# --- Helper Functions ---

def prepare_generator_input(image_data, glyph_size, glyphs_per_image):
  base = random.randint(0, glyphs_per_image - 1)
  image_width = glyph_size[1]
  return image_data[:,:,:,base * image_width:(base + 1) * image_width]

def reshape_real_data(real_data, glyph_size, glyphs_per_image):
  return real_data[:, :, :, 0:(glyphs_per_image * glyph_size[1])]

def reshape_generated_data(generated_output, glyph_size, glyphs_per_image):
  # generated_shape = generated_output.shape
  return generated_output[:, :, :, 0:(glyphs_per_image * glyph_size[1])]
  # # Flatten the output, then take only letters A-Z (64 x 26 = 1664) -- Ignore the dead space
  # return generated_output.reshape(
  #   generated_shape[0],
  #   generated_shape[1],
  #   glyph_size[0],
  #   glyph_size[1] * 32
  # )[:, :, :, 0:glyph_size[1] * glyphs_per_image]

def record_losses(loss_plot, losses):
  record = {}
  for key in losses.keys():
    record[key] = losses[key][-1]
  loss_plot.update(record)
  loss_plot.send()

def prepare_static_test(data_loader, options):
  real_test = None
  static_test = None
  glyph_size, glyphs_per_image, data_type = itemgetter('glyph_size', 'glyphs_per_image', 'data_type')(options)

  for data in data_loader:
    print('=== Test Font ===')
    real_test = reshape_real_data(data, glyph_size, glyphs_per_image)
    static_test = prepare_generator_input(data, glyph_size, glyphs_per_image)[0:1].type(data_type)
    print('=== Initial Output ===')

    break

  return real_test, static_test
