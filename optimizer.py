from torch.optim import Optimizer, LBFGS
from image.container import ImageContainer

def make_optimizer(
  target_img: ImageContainer
) -> Optimizer:
  # Note:
  # Since the space of loss function is quite flat,
  # typical learning algorithm such as Adam is not appropriate so much.
  # So using classic LBFGS is appropriate in this case..
  return LBFGS([target_img.parameter])