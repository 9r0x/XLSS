# XLSS

# Install

```shell
pip install tensorflow yapf jupyterlab numpy pillow matplotlib
```

# Training

## Prepare data

```
Usage: python3 to_patch.py -n <num_samples> -c <input_path1> <output_dir1> <size1> <input_path2> <output_dir2> <size2> ...
                      -n: Number of samples to preprocess from each directory
                      -c: Clean the output directory before preprocessing
Example 1: python3 to_patch.py -n 200 -c \
        ./data/Flickr2K/Flickr2K_HR ./data/minibatch/HR 100 \
        ./data/Flickr2K/Flickr2K_LR_bicubic/X2 ./data/minibatch/LR 50

Example 2: python3 to_patch.py \
        ./data/Flickr2K/Flickr2K_LR_bicubic/X2/000001x2.png ./data/minibatch/test 50
```

```shell
python3 to_patch.py -n 500 -c \
        ./data/Flickr2K/Flickr2K_HR ./data/minibatch/HR 100 \
        ./data/Flickr2K/Flickr2K_LR_bicubic/X2 ./data/minibatch/LR 50

python3 train.py ./data/minibatch/LR ./data/minibatch/HR
```

# Testing

```shell
python3 xlss.py ./data/minibatch/test ./data/minibatch/test_out
```
