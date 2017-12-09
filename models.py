from torchvision.models import vgg19
from torch.nn import Module, Sequential, MSELoss
from torch import Tensor, mm
from torch.autograd import Variable

from image_container import ImageContainer

class ContentLoss(Module):
  def __init__(self, target, weight):
    # So to prevent the gradient flow w.r.t target, detach is necessary.
    super().__init__()
    self.target = target.detach() * weight
    self.weight = weight
    self.criterion = MSELoss()

  def forward(self, x):
    self.loss = self.criterion(x * self.weight, self.target)
    return x

  def backward(self):
    self.loss.backward(retain_graph = True)
    return self.loss

def gram_matrix(x: Tensor):
  _, b, c, d = x.size()
  f_hat = x.view(b, c * d)
  G = mm(f_hat, f_hat.t())
  return G.div(b * c * d)

class StyleLoss(Module):
  def __init__(self, target, weight):
    super().__init__()
    self.target = target.detach() * weight
    self.criterion = MSELoss()
    self.weight = weight
  
  def forward(self, x):
    self.output =  x.clone()
    G = gram_matrix(x)
    G = G.mul(self.weight)
    self.loss = self.criterion(G, self.target)
    return self.output

  def backward(self):
    self.loss.backward(retain_graph = True)
    return self.loss


def make_network(
  content_img: ImageContainer,
  style_img: ImageContainer,
  content_layers_indexes = [7],
  style_layers_indexes = [0, 2, 5, 7],
  content_weight = 1,
  style_weight = 3000
):
  vgg19_pretrained = vgg19(True).features
  net = Sequential()
  content_losses = []
  style_losses = []
  for i, module in enumerate(vgg19_pretrained):
    net.add_module(f'layer_{i}', module)
    if i in style_layers_indexes:
      target = net(style_img.variable).clone()  # TODO: INVESTIGATE
      target_gram = gram_matrix(target)
      style_loss = StyleLoss(target_gram, style_weight)
      net.add_module(f'style_loss_{i}', style_loss)
      style_losses.append(style_loss)

    if i in content_layers_indexes:
      target = net(content_img.variable).clone()  # TODO: INVESTIGATE
      content_loss = ContentLoss(target, content_weight)
      net.add_module(f'content_loss_{i}', content_loss)
      content_losses.append(content_loss)

  return net, content_losses, style_losses