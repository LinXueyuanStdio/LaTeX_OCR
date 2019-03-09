# Im2Latex

Seq2Seq model with Attention + Beam Search for Image to LaTeX.

Similar to [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) and [Harvard's paper and dataset](http://lstm.seas.harvard.edu/latex/).

Check the [blog post](https://guillaumegenthial.github.io/image-to-latex.html).

## 1. Install

Any questoin when you run `make`, plz read the file `./makefile` for details.

If you have installed, it means you are able to build the images from formulas by running latex render.

Install pdflatex (latex to pdf) and ghostsript + [magick](https://www.imagemagick.org/script/install-source.php
) (pdf to png) on Linux

```
make install-linux
```

(takes a while ~ 10 min, installs from source)

On Mac, assuming you already have a LaTeX distribution installed, you should have pdflatex and ghostscript installed, so you just need to install magick. You can try

```
make install-mac
```

## 2. Getting Started

### Training on the small dataset

it only needs 2 mins for building 100 images from the formulas you provide in `./data/small.formulas/`.

We provide a small dataset just to check the pipeline. To build the images, train the model and evaluate

```
make small
```

You should observe that the model starts to produce reasonable patterns of LaTeX after a few minutes.

### Training on the full dataset

It needs hours for building 70,000+ images

If you already did `make build` you can just train and evaluate the model with the following commands

```
make train
make eval
```

Or, to build the images from the formulas, train the model and evaluate, run

```
make full
```


## 3. Details for you to debug

### Training on the small dataset

1. Build the images from the formulas, write the matching file and extract the vocabulary. __Run only once__ for a dataset

```
python build.py
```

or

```
python build.py --data=configs/data_small.json --vocab=configs/vocab_small.json
```

1. Train

```
python train.py
```

or

```
python train.py --data=configs/data_small.json --vocab=configs/vocab_small.json --training=configs/training_small.json --model=configs/model.json --output=results/small/
```

1. Evaluate the text metrics

```
python evaluate_txt.py
```

or

```
python evaluate_txt.py --results=results/small/
```

1. Evaluate the image metrics

```
python evaluate_img.py
```

or

```
python evaluate_img.py --results=results/small/
```


### Training on the full dataset

1. Build the images from the formulas, write the matching file and extract the vocabulary. __Run only once__ for a dataset
```
python build.py --data=configs/data.json --vocab=configs/vocab.json
```

2. Train
```
python train.py --data=configs/data.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/full/
```

3. Evaluate the text metrics
```
python evaluate_txt.py --results=results/full/
```

4. Evaluate the image metrics
```
python evaluate_img.py --results=results/full/
```

## 4. Cheatsheet

### Visualize with tensorboard 可视化

for small dataset:
```
cd results/small
tensorboard --log-dir ./
```

for full dataset:

```
cd results/full
tensorboard --log-dir ./
```

## 5. Problems

### 5.1 理想情况下输出数据应该按照一定规规律随着输入数据的变化而变化的，但是训练到最后，无论输入数据是多，输出数据都是一个数值，loss在来回跳动，没有减小。

遇到了这个问题，我的loss值最开始是在比较大的值上一直无法收敛，查看网络权值梯度，最开始的梯度返回已经是e-3级别了，因此网络基本没调整。

目前知道的方法有：

1. 初始化不对，查看每层网络的输入方差，如果方差越来越小，可以判定为初始化不对，而你的初始化也是采用的基本的高斯函数，有很大可能是初始化的问题；

2. 采用BN（批量标准化），可以稍微降低初始化对网络的影响；

3. 你的预测输出标签，而不是概率。如果你的数据是非常倾斜，1:20什么的，基本上不管怎么预测，预测标签都会是0。

4. 你的模型层数比较多，激活函数都是一样的。可以试试两层、三层的，效果如何。也可以换个激活函数试试。

最后解决: 这不是过拟合，这™是欠拟合，训练多个epoch就行。

### 5.2 attention 的可视化

想要这种效果：[https://github.com/ritheshkumar95/im2latex-tensorflow](https://github.com/ritheshkumar95/im2latex-tensorflow)

用全局变量做