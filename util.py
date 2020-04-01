import matplotlib.pyplot as plot
import torchvision.transforms as transforms

def show_grayscale_image(image):
  plot.imshow(transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3)
  ])(image))
  plot.axis('off')
  plot.show()