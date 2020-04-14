import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))
#TODO: Swap to syspath.append(path.dirname(path.dirname(path.realpath(file))))

import unittest

from data.font_dataset import FontDataset

class TestFontDatasets(unittest.TestCase):
  def test_cannot_create_invalid_font_dataset(self):
    with self.assertRaises(AssertionError):
      FontDataset('does_not_exist')

  def test_can_create_font_dataset(self):
    dataset = FontDataset(abspath(join(dirname(__file__), 'test_datasets/valid')))
    self.assertEqual(1, len(dataset))

  def test_length_of_empty_folder(self):
    dataset = FontDataset(abspath(join(dirname(__file__), 'test_datasets/empty')))
    self.assertEqual(0, len(dataset))
