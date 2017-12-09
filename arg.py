from argparse import ArgumentParser

def build_argument_parser():
  parser = ArgumentParser(
    description='Does the Nerual-Style-Transfer to target arbitrary images in vanilla implementation'
  )

  parser.add_argument('--content_image_path', '-c', required = True, type = str)
  parser.add_argument('--style_image_path', '-s', required = True, type = str)
  parser.add_argument('--output_image_path', '-o', required = True, type = str)

  parser.add_argument('--iter_epochs', '-i', default = 200, type = int)
  parser.add_argument('--content_weight', '-a', default = 1, type = int)
  parser.add_argument('--style_weight', '-b', default = 3000, type = int)

  return parser