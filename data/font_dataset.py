import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image

class FontData():
  def __init__(self, font_name, font_path, image=None):
    self.font_name = font_name
    self.font_path = font_path
    self.image = None

  def load_data(self, loader):
    if self.image == None:
      self.image = loader(self.font_path)
    return self.image

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

    font = self.fonts[_index]
    font_data = font.load_data(image_loader)

    #TODO: Our images are 64 x 1664 -- This should be a parameter/configuration option.
    transform = transforms.Compose([
      transforms.Grayscale(), # Drop to 1 channel
      transforms.Resize((16, 416)),
      transforms.ToTensor()
    ])

    return transform(font_data)

  def load_font_filenames(self, root_dir):
    font_images = []
    assert os.path.isdir(root_dir), '%s is not a valid directory!' % root_dir

    for root, _, filenames in sorted(os.walk(root_dir)):
      for filename in filenames:
        font_images.append(FontData(filename, os.path.join(root, filename)))

    return font_images

# Helper Functions

def image_loader(path):
  # Convert to greyscale
  return Image.open(path).convert('RGB')