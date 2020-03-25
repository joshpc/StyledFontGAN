def train(D, G, D_optimizer, G_optimizer, batch_size, epoch_count):
  """
  Main training loop for StyledFontGAN.
  """

  #TODO: Get the data loader
  data_loader = None

  for epoch in range(epoch_count):
    epoch_start_time = time.time()
    train_epoch()
    print("{} --- G: {:4} | D: {:.4} | GP: {:.4} | GNorm: {:.4} --- Total time: {}".format(int(epoch + 1), losses['G'][-1], losses['D'][-1], losses['GP'][-1], losses['gradient_norm'][-1], (time.time() - epoch_start_time)))

def train_epoch():
  steps = 0

  return None

def train_generator():
  return None

def train_discriminator():
