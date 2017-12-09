from image_container import ImageContainer
from models import make_network
from optimizer import make_optimizer

# TODO:
#   You can change these pathes or make this code to accept
#   command line parameters so you can specify arbitrary parameters
CONTENT_IMAGE_PATH = './images/daughter.jpg'
STYLE_IMAGE_PATH = './images/mona_lisa.jpg'
NEW_IMAGE_PATH = './images/out3.jpg'
ITER_EPOCHS = 200

content_img = ImageContainer(CONTENT_IMAGE_PATH)
style_img = ImageContainer(STYLE_IMAGE_PATH)

# The original paper shows that you can use random noise as an input however,
# to make learning easier, here, we use exactly the same image as content.
target_img = ImageContainer(CONTENT_IMAGE_PATH)

net, content_losses, style_losses = make_network(
  content_img,
  style_img
)

optimizer = make_optimizer(
  target_img
)

i = 0
while i < ITER_EPOCHS:
  def closure():
    global i
    i += 1
    # Do the forward pass
    target_img.clamp_()
    target_img.feed(net)

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
      print(f'Iter: {i} / {ITER_EPOCHS}, Loss: {total_loss.data[0]}, C: {total_content_loss.data[0]}, S: {total_style_loss.data[0]}')
    return total_loss
  optimizer.step(closure)
target_img.save(NEW_IMAGE_PATH)