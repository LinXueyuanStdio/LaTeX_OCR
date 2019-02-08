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
