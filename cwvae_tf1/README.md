# Clockwork Variational Autoencoders (CW-VAE)

Vaibhav Saxena, Jimmy Ba, Danijar Hafner

<img src="https://danijar.com/asset/cwvae/header.gif" height="300">

This project provides the open source implementation of the [Clockwork Variational Autoencoder](https://arxiv.org/abs/2102.09532). Please visit the [project homepage](https://danijar.com/project/cwvae/) for more illustrations and details.

If you find this open source release useful, please reference in your paper:

```
@article{saxena2021clockworkvae,
  title={Clockwork Variational Autoencoders}, 
  author={Saxena, Vaibhav and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2102.09532},
  year={2021},
}
```

## Training

1. Update the following config parameters in configs/minerl.yml: 
  * `data_root` : directory where the data will be downloaded and extracted. (For MineRL Navigate, this will take ~5GB on the disk.)
  * `log_root` : root directory for storing logs
  * any other model params (such as `levels`, `tmp_abs_factor`)

2. Run the following:
```
python3 train.py --config ./configs/<dataset>.yml
```
where you may replace `dataset` with `minerl`, `mazes`, `mmnist`.

## Model Evaluation

For MineRL Navigate:
```
python3 eval.py --model-dir=/path/to/saved/model --open-loop-ctx=36 --seq-len=500
```

For Moving MNIST:
```
python3 eval.py --model-dir=/path/to/saved/model --open-loop-ctx=36 --seq-len=1000
```

For GQN Mazes:
```
python3 eval.py --model-dir=/path/to/saved/model --open-loop-ctx=36 --seq-len=300
```

You may set the following args:
* `--open-loop-ctx` : number of input frames (while maintaining compatibility b/w number of levels and temporal abstraction factor)
* `--seq-len` : total sequence length of an example
* `--data-root` : path to file/dir containing data
* `--use-obs` : string of `T`/`F`s denoting whether or not to use observations at a level while building context, e.g. `TTF` for a 3-level model will ignore the observations at the top level
* `--num-examples` : number of examples to evaluate
* `--num-samples` : number of samples to generate per example
