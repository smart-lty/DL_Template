# A Simple Template for Deep Learning.

Here is a template for deep learning projects. Feel free to change the codes.

Generally, coding from a template will alliviate your burden and make you more concentrate on some key points (such as model architecture). Therefore, I release a template of my common usage, where some key points of the projects are omitted(see codes).

Now I am working on improving this projects. Feel free to contact me at tianyu_liu@mail.ustc.edu.cn with any questions.

A blog of this template can be found at my [personal website](https://smart-lty.github.io/tech/dl_template/).

## Installation and Preparation
I recommand to create a conda environment for a unique project.

```shell
conda create -n env1 python=3.8
conda activate env1
```
Though many Python versions can be used, I recommand to use python 3.8.

## Run
Just a line of commands to start training and evaluation!
```shell
python src/main.py --gpu 0
```

Other parameters are specified in `util.py`.