from PIL import Image
from torch import unsqueeze
from torch.autograd import Variable
from torchvision.transforms import Compose, Scale, ToTensor, ToPILImage

class ImageContainer:
  def __init__(self, path, imsize=256):
    image = Image.open(path)
    transformer = Compose([
      Scale([imsize, imsize]),  # Ensures square image
      ToTensor()
    ])
    self.variable = Variable(unsqueeze(transformer(image), 0))
    self.imsize = imsize

  def save(self, path):
    self.image.save(path)

  @property
  def image(self):
    return ToPILImage()(self.variable.data.view(3, self.imsize, self.imsize))