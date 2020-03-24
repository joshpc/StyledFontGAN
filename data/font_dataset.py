import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class FontData():
  def __init__(self, font_name, font_path):
    self.font_name = font_name
    self.font_path = font_path

  def load_data(self, loader):
    self.image = loader(self.font_path)

  def __repr__(self):
    return "<FontData font_name: %s>" % self.font_name

class FontDataset(Dataset):
  """The Font Dataset."""

  def __init__(self, root_dir, number_of_characters_per_image=10, transform=None):
    """
    """
    self.fonts = self.load_font_filenames(root_dir)
    self.root_dir = root_dir
    self.number_of_characters_per_image = number_of_characters_per_image
    self.transform = transform

  def __len__(self):
    return len(self.fonts)

  def __getitem__(self, index):
    _index = index
    if torch.is_tensor(_index):
      _index = _index.tolist()

    return self.fonts[_index]

  def load_font_filenames(self, root_dir):
    font_images = []
    assert os.path.isdir(root_dir), '%s is not a valid directory!' % root_dir

    for root, _, filenames in sorted(os.walk(root_dir)):
      for filename in filenames:
        font_images.append(FontData(filename, os.path.join(root, filename)))

    return font_images

# Helper Functions

def image_loader(path):
  return Image.open(path).convert('RGB')