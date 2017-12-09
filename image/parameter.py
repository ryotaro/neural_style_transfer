from torch import unsqueeze
from torch.nn import Parameter

from .container import ImageContainer

class ImageParameter(ImageContainer):
  def __init__(self, path, imsize=256):
    super().__init__(path, imsize)
    self.parameter = Parameter(self.variable.data)

  def feed(self, net):
    return net(self.parameter)

  def clamp_(self):
    self.parameter.data.clamp_(0, 1)