import time

def train(D, G, D_optimizer, G_optimizer, batch_size, epoch_count):
  """
  Main training loop for StyledFontGAN.
  """

  #TODO: Get the data loader
  data_loader = None

  for epoch in range(epoch_count):
    epoch_start_time = time.time()
    train_epoch(D, G, D_optimizer, G_optimizer, batch_size)
    # print("{} --- G: {:4} | D: {:.4} | GP: {:.4} | GNorm: {:.4} --- Total time: {}".format(int(epoch + 1), losses['G'][-1], losses['D'][-1], losses['GP'][-1], losses['gradient_norm'][-1], (time.time() - epoch_start_time)))

def train_epoch(D, G, D_optimizer, G_optimizer, batch_size):
  steps = 0

  #TODO: Get the data_loader
  # data_loader = None
  # for data, labels in data_loader:
  #   if len(data) % batch_size != 0:
  #     continue

  steps += 1

  train_discriminator(D, G, D_optimizer)

  # TODO: Parameterize
  if steps % 5 == 0:
    train_generator(D, G, G_optimizer)

  return None

def train_generator(D, G, G_optimizer):
  """
  Executes one interation of training for the generator.

  No return value.
  """
  G_optimizer.zero_grad()

  # Prepare our data
  generator_input = None
  generated_data = G(generator_input)

  # Forward pass with the discriminator
  discriminator_loss = D(generated_data)

  # Update the loss
  loss = -discriminator_loss.mean()
  loss.backward()

  G_optimizer.step()

def train_discriminator(D, G, D_optimizer):
  """
  Executes one iteration of training for the discriminator.

  No return value.
  """
  D_optimizer.zero_grad()

  # Prepare the data
  generator_input = None
  generated_data = G(generator_input)

  real_input = None
  real_loss = D(real_input)
  generated_loss = D(generated_data)

  # Calculate loss
  #TODO: How should the loss be structured?

  # Calculate the Wasserstein Distance
  loss = generated_loss.mean() - real_loss.mean()
  loss.backward()

  D_optimizer.step()