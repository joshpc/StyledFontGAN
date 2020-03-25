import torch

class FontDataLoader():
  def __init__(self, dataset, batch_size, shuffle=True):
    self.data_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle
    )

  def __iter__(self):
    self.data_loader_iterator = iter(self.data_loader)
    return self

  def __next__(self):
    return next(self.data_loader_iterator)
