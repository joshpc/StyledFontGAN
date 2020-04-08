import torch

class FontDataLoader():
  def __init__(self, dataset, sampler, batch_size):
    self.data_loader = torch.utils.data.DataLoader(
      dataset,
      sampler=sampler,
      batch_size=batch_size
    )

  def __iter__(self):
    self.data_loader_iterator = iter(self.data_loader)
    return self

  def __next__(self):
    return next(self.data_loader_iterator)
