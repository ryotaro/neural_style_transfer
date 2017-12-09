from typing import AnyStr
from image.container import ImageContainer
from image.parameter import ImageParameter
from optimizer import make_optimizer
from models import make_network

class IterationAccumulator(int): pass

def train(
  content_image_path: AnyStr,
  style_image_path: AnyStr,
  iter_epochs: int = 200,
  content_weight: int = 1,
  style_weight: int = 3000,
  ) -> ImageContainer:

  content_image = ImageContainer(content_image_path)
  style_image = ImageContainer(style_image_path)

  # The original paper shows that you can use random noise as an input however,
  # to make learning easier, here, we use exactly the same image as content.
  target_image = ImageParameter(content_image_path)

  net, content_losses, style_losses = make_network(
    content_image,
    style_image,
    content_weight = content_weight,
    style_weight = style_weight
  )

  optimizer = make_optimizer(target_image)

  i = IterationAccumulator()
  while i < iter_epochs:
    def closure():
      nonlocal i
      # Do the forward pass
      target_image.feed(net)

      # Do the BP
      optimizer.zero_grad()
      total_content_loss = .0
      total_style_loss = .0
      for content_loss_layer in content_losses:
        total_content_loss += content_loss_layer.backward()
      for style_loss_layer in style_losses:
        total_style_loss += style_loss_layer.backward()

      total_loss = total_content_loss + total_style_loss
      if i % 10 == 0:
        print(f'Iter: {i} / {iter_epochs}, Loss: {total_loss.data[0]}, C: {total_content_loss.data[0]}, S: {total_style_loss.data[0]}')
      i += 1
      return total_loss
    optimizer.step(closure)
    target_image.clamp_()
  return target_image
