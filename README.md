# Vanilla Neural Style Transfer Implementation Written in PyTorch
## Overview
This script generates a new image from given two images using [Neural Style Transfer algorithm](https://arxiv.org/abs/1508.06576).
You can generate an artistic image by specifying base content image and target style image which you want to apply the style to the content image.

![]()

## Requirement
Python 3.6

## Installation
```bash
# It installs PyTorch and its dependencies.
# If you are using a virtual environment tool, create & activate the environment beforehand.
$ pip3 install -r requirements.txt
```

## How to Use
Execute `main.py` with following command-line parameters.


## Parameters

| Option | Description |
| ------ | ----------- |
| `-h, --help` | Show usage |
| `-c CONTENT_IMAGE_PATH` |  (REQUIRED) path for content image (jpeg) |
| `-s STYLE_IMAGE_PATH` | (REQUIRED) path for style image (jpeg) |
| `-o OUTPUT_IMAGE_PATH` |  (REQUIRED) path for output image (jpeg) |
| `-i ITER_EPOCHS` |  (OPTIONAL) how many epochs to iterate in train time. DEFAULT=200 |
| `-a CONTENT_WEIGHT` | (OPTIONAL) how much it does penalize for content image. DEFAULT=1 |
| `-b STYLE_WEIGHT` | (OPTIONAL) how muc hit does penalize for style image. DEFAULT=3000|

## Example
```
$ time python3 main.py -c ./content_image.jpg  -s ./style_image.jpg -o ./out.jpg -i 250 -b 10000
Iter: 0 / 250, Loss: 1085.4854736328125, C: 0.0, S: 1085.4854736328125
Iter: 10 / 250, Loss: 169.38926696777344, C: 0.5092644691467285, S: 168.8800048828125
Iter: 20 / 250, Loss: 81.15397644042969, C: 0.7630603909492493, S: 80.39091491699219
Iter: 30 / 250, Loss: 38.373504638671875, C: 0.9629819393157959, S: 37.4105224609375
Iter: 40 / 250, Loss: 24.216154098510742, C: 1.0101701021194458, S: 23.205984115600586
Iter: 50 / 250, Loss: 17.071678161621094, C: 1.0389184951782227, S: 16.032760620117188
Iter: 60 / 250, Loss: 13.898388862609863, C: 1.0392717123031616, S: 12.85911750793457
Iter: 70 / 250, Loss: 10.952510833740234, C: 1.067094087600708, S: 9.885416984558105
Iter: 80 / 250, Loss: 9.188867568969727, C: 1.0860315561294556, S: 8.102835655212402
Iter: 90 / 250, Loss: 7.020061492919922, C: 1.1368275880813599, S: 5.883234024047852
Iter: 100 / 250, Loss: 6.455777645111084, C: 1.1681337356567383, S: 5.287643909454346
Iter: 110 / 250, Loss: 5.056793212890625, C: 1.2013438940048218, S: 3.8554491996765137
Iter: 120 / 250, Loss: 4.732418060302734, C: 1.197936773300171, S: 3.5344810485839844
Iter: 130 / 250, Loss: 4.13242769241333, C: 1.2011055946350098, S: 2.9313220977783203
Iter: 140 / 250, Loss: 3.8782098293304443, C: 1.180586338043213, S: 2.6976234912872314
Iter: 150 / 250, Loss: 3.500723361968994, C: 1.1636804342269897, S: 2.337042808532715
Iter: 160 / 250, Loss: 3.3245725631713867, C: 1.1438530683517456, S: 2.1807193756103516
Iter: 170 / 250, Loss: 3.0777363777160645, C: 1.13227379322052, S: 1.9454624652862549
Iter: 180 / 250, Loss: 2.899379014968872, C: 1.116103172302246, S: 1.783275842666626
Iter: 190 / 250, Loss: 2.6403286457061768, C: 1.102752685546875, S: 1.5375759601593018
Iter: 200 / 250, Loss: 2.5092577934265137, C: 1.0929688215255737, S: 1.4162888526916504
Iter: 210 / 250, Loss: 2.278076648712158, C: 1.091755747795105, S: 1.1863207817077637
Iter: 220 / 250, Loss: 2.1692404747009277, C: 1.076847791671753, S: 1.0923925638198853
Iter: 230 / 250, Loss: 2.0064969062805176, C: 1.0773088932037354, S: 0.9291881322860718
Iter: 240 / 250, Loss: 1.9337048530578613, C: 1.0694180727005005, S: 0.8642867207527161
Iter: 250 / 250, Loss: 1.7941423654556274, C: 1.0674238204956055, S: 0.726718544960022
      550.72 real      1482.52 user       133.15 sys

# Now the generated image is saved to out.jpg!
```

## Warning
No CUDA support yet.. (ごめん)

## License
MIT License

Copyright (c) 2017 Ryotaro IKEDA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.