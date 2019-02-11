# Im2Latex

Seq2Seq model with Attention + Beam Search for Image to LaTeX, similar to [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) and [Harvard's paper and dataset](http://lstm.seas.harvard.edu/latex/).

Check the [blog post](https://guillaumegenthial.github.io/image-to-latex.html).

## Install

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

## Getting Started

We provide a small dataset just to check the pipeline. To build the images, train the model and evaluate

```
make small
```

You should observe that the model starts to produce reasonable patterns of LaTeX after a few minutes.


## Data

We provide the pre-processed formulas from [Harvard](https://zenodo.org/record/56198#.V2p0KTXT6eA) but you'll need to produce the images from those formulas (a few hours on a laptop).

```
make build
```

Alternatively, you can download the [prebuilt dataset from Harvard](https://zenodo.org/record/56198#.V2p0KTXT6eA) and use their preprocessing scripts found [here](https://github.com/harvardnlp/im2markup)


## Training on the full dataset

If you already did `make build` you can just train and evaluate the model with the following commands

```
make train
make eval
```

Or, to build the images from the formulas, train the model and evaluate, run

```
make full
```


## Details

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

(To get more information on the arguments, run)

```
python file.py --help
```

## Problems

### 理想情况下输出数据应该按照一定规规律随着输入数据的变化而变化的，但是训练到最后，无论输入数据是多，输出数据都是一个数值，loss在来回跳动，没有减小。

遇到了这个问题，我的loss值最开始是在比较大的值上一直无法收敛，查看网络权值梯度，最开始的梯度返回已经是e-3级别了，因此网络基本没调整。

目前知道的方法有：

初始化不对，查看每层网络的输入方差，如果方差越来越小，可以判定为初始化不对，而你的初始化也是采用的基本的高斯函数，有很大可能是初始化的问题；

采用BN（批量标准化），可以稍微降低初始化对网络的影响；

1、你的预测输出标签，而不是概率。如果你的数据是非常倾斜，1:20什么的，基本上不管怎么预测，预测标签都会是0。

2、你的bias项一直都是0，tf.Variable(tf.constant(0.0, shape=[self.num_hidden]))。这也许是一个可能。

3、你的模型层数比较多，激活函数都是一样的。可以试试两层、三层的，效果如何。也可以换个激活函数试试。

