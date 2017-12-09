from train import train

# TODO:
#   You can change these pathes or make this code to accept
#   command line parameters so you can specify arbitrary parameters
CONTENT_IMAGE_PATH = './images/daughter.jpg'
STYLE_IMAGE_PATH = './images/mona_lisa.jpg'
NEW_IMAGE_PATH = './images/out3.jpg'
ITER_EPOCHS = 200

target_image = train(
  CONTENT_IMAGE_PATH,
  STYLE_IMAGE_PATH,
  iter_epochs = 200,
  content_weight = 1.0,
  style_weight = 5000.
)

target_image.save(NEW_IMAGE_PATH)