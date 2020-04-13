from operator import itemgetter

import torch
from torch.autograd import grad as torch_grad

def wasserstein_loss(D, real_data, generated_data, losses, options):
  real_loss = D(real_data)
  generated_loss = D(generated_data)

  batch_size, data_type = itemgetter('batch_size', 'data_type')(options)
  gradient_penalty_weight = 10

  # Calculate gradient penalty
  gradient_penalty = calculate_gradient_penalty(D, real_data, generated_data, batch_size, gradient_penalty_weight, losses, data_type)
  losses['GP'].append(gradient_penalty.data)

  # Calculate the Wasserstein Distance.
  loss = generated_loss.mean() - real_loss.mean() + gradient_penalty
  losses['Generated'].append(generated_loss.mean().data)
  losses['Real'].append(real_loss.mean().data)
  losses['D'].append(loss.data)

  return loss

def min_max_loss(D, G, real_data, generated_data, losses, options):
  # Forward pass with the discriminator
  discriminator_loss = D(generated_data)

  # Update the loss. We're trying to fool the discriminator to say '1, this is real'
  loss = -discriminator_loss.mean()

  return loss

def l1_loss(D, G, real_data, generated_data, losses, options):
  """
  Performs the L1 loss between the generated data and the real data.

  It is expected that both `real_data` and `generated_data` are of the same shape.
  """
  return torch.nn.L1Loss()(generated_data, real_data)


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