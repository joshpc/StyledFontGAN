import time
import random

import torch
from torch.autograd import grad as torch_grad

# Visualization
from livelossplot import PlotLosses
from util import show_grayscale_image

def train(D, G, D_optimizer, G_optimizer, batch_size, epoch_count, data_loader, data_type, glyph_size, glyphs_per_image):
  """
  Main training loop for StyledFontGAN.
  """
  liveloss = PlotLosses()
  losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'Generated': [], 'Real': []}

  for data in data_loader:
    print('=== Test Font ===')
    static_test = prepare_generator_input(data, glyph_size, glyphs_per_image)[0:1].type(data_type)
    show_grayscale_image(static_test[0].cpu())
    print('=== Initial Output ===')
    show_grayscale_image(G(static_test)[0].cpu())
    break

  for epoch in range(epoch_count):
    epoch_start_time = time.time()

    # print("{} - Starting".format(int(epoch)))
    train_epoch(D, G, D_optimizer, G_optimizer, batch_size, data_loader, data_type, glyph_size, glyphs_per_image, losses)
    # print("{} --- G: {:4} | D: {:.4} | GP: {:.4} | gradient_norm: {:.4} | Gen: {:.4} | Real: {:.4} --- Total time: {}".format(int(epoch + 1), losses['G'][-1], losses['D'][-1], losses['GP'][-1], losses['gradient_norm'][-1], losses['Generated'][-1], losses['Real'][-1], (time.time() - epoch_start_time)))

    liveloss.update({
      'G': losses['G'][-1],
      'D': losses['D'][-1],
      'GP': losses['GP'][-1],
      'gradient_norm': losses['gradient_norm'][-1],
      'Gen': losses['Generated'][-1],
      'Real': losses['Real'][-1],
    })
    liveloss.send()

    show_grayscale_image(G(static_test)[0].cpu())

def train_epoch(D, G, D_optimizer, G_optimizer, batch_size, data_loader, data_type, glyph_size, glyphs_per_image, losses):
  steps = 0
  gradient_penalty_weight = 10

  for data in data_loader:
    if len(data) % batch_size != 0:
      continue
    data = data.type(data_type)

    steps += 1

    train_discriminator(D, G, D_optimizer, data, glyph_size, glyphs_per_image, losses, batch_size, gradient_penalty_weight, data_type)

    # TODO: Parameterize
    if steps % 5 == 0:
      train_generator(D, G, G_optimizer, data, glyph_size, glyphs_per_image, losses)

def train_generator(D, G, G_optimizer, data, glyph_size, glyphs_per_image, losses):
  """
  Executes one interation of training for the generator.

  No return value.
  """
  G_optimizer.zero_grad()

  # Prepare our data. We only use the letter A to seed this entire process.
  generator_input = prepare_generator_input(data, glyph_size, glyphs_per_image)
  generated_data = G(generator_input)

  # Forward pass with the discriminator
  discriminator_loss = D(reshape_generated_data(generated_data, glyph_size, glyphs_per_image))

  # Update the loss. We're trying to fool the discriminator to say '1, this is real'
  loss = -discriminator_loss.mean()
  loss.backward()
  losses['G'].append(loss.data)

  G_optimizer.step()

def train_discriminator(D, G, D_optimizer, data, glyph_size, glyphs_per_image, losses, batch_size, gradient_penalty_weight, data_type):
  """
  Executes one iteration of training for the discriminator.

  No return value.
  """
  D_optimizer.zero_grad()

  # Prepare the data
  generator_input = prepare_generator_input(data, glyph_size, glyphs_per_image)
  generated_data = reshape_generated_data(G(generator_input), glyph_size, glyphs_per_image)

  # show_grayscale_image(reshape_generated_data(generated_data, glyph_size, glyphs_per_image)[0].cpu())
  # show_grayscale_image(data[0].cpu())
  real_loss = D(data)
  generated_loss = D(generated_data)

  # Calculate gradient penalty
  gradient_penalty = calculate_gradient_penalty(D, data, generated_data, batch_size, gradient_penalty_weight, losses, data_type)
  losses['GP'].append(gradient_penalty.data)

  # Calculate the Wasserstein Distance.
  loss = generated_loss.mean() - real_loss.mean()
  loss.backward()
  losses['Generated'].append(generated_loss.mean().data)
  losses['Real'].append(real_loss.mean().data)
  losses['D'].append(loss.data)

  D_optimizer.step()

# Helper Functions

def prepare_generator_input(image_data, glyph_size, glyphs_per_image):
  base = random.randint(0, glyphs_per_image - 1)
  image_width = glyph_size[1]
  return image_data[:,:,:,base * image_width:(base + 1) * image_width]

def reshape_generated_data(generated_output, glyph_size, glyphs_per_image):
  generated_shape = generated_output.shape
  return generated_output
  # # Flatten the output, then take only letters A-Z (64 x 26 = 1664) -- Ignore the dead space
  # return generated_output.reshape(
  #   generated_shape[0],
  #   generated_shape[1],
  #   glyph_size[0],
  #   glyph_size[1] * 32
  # )[:, :, :, 0:glyph_size[1] * glyphs_per_image]

def calculate_gradient_penalty(D, real_data, generated_data, batch_size, gradient_penalty_weight, losses, data_type):
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1).expand_as(real_data).type(data_type)

    # 'interpolated' is x-hat
    interpolated = (alpha * real_data.data + (1 - alpha) * generated_data.data).type(data_type)
    interpolated.requires_grad = True

    # Calculate probability of interpolated examples
    probability_interpolated = D(interpolated)

    # TODO: Clean up?
    gradients = torch_grad(outputs=probability_interpolated,
                           inputs=interpolated,
                           grad_outputs=torch.ones(
                               probability_interpolated.size()).type(data_type),
                           create_graph=True,
                           retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gradient_penalty_weight * ((gradients_norm - 1) ** 2).mean()