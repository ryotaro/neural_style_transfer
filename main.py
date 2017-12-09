from train import train
from arg import build_argument_parser

parser = build_argument_parser()
args = parser.parse_args()

target_image = train(
  args.content_image_path,
  args.style_image_path,
  iter_epochs = args.iter_epochs,
  content_weight = args.content_weight,
  style_weight = args.style_weight
)

target_image.save(args.output_image_path)