from PIL import Image
from torch import unsqueeze
from torch.nn import Parameter
from torch.autograd import Variable
from torchvision.transforms import Compose, Scale, ToTensor, ToPILImage, Normalize

# TODO:
# Separate functionality: Fixed target images & an image to be fed
class ImageContainer:
  def __init__(self, path, normalize_for_vgg19=False, imsize=256):
    image = Image.open(path)
    transformer = Compose([
      Scale([imsize, imsize]),  # Ensures square image
      ToTensor()
    ])
    self.variable = Variable(unsqueeze(transformer(image), 0))
    self.parameter = Parameter(self.variable.data)
    self.imsize = imsize

  def feed(self, net):
    return net(self.parameter)

  def save(self, path):
    self.image.save(path)

  def clamp_(self):
    self.parameter.data.clamp_(0, 1)

  @property
  def image(self):
    return ToPILImage()(self.variable.data.view(3, self.imsize, self.imsize))