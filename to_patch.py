import os, sys, argparse
from collections import defaultdict
from multiprocessing import Pool
from PIL import Image


class ImageProcessor:

    def __init__(self, ipath, opath, size, num_samples):
        self.ipath = ipath
        self.opath = opath
        self.size = size
        self.num_samples = num_samples
        self.suffix = 'x2'

        if not os.path.exists(self.opath):
            os.makedirs(self.opath)

    def crop_to_patches(self, image):
        img_width, img_height = image.size
        ncols = img_width // self.size
        nrows = img_height // self.size
        image = image.crop((0, 0, ncols * self.size, nrows * self.size))

        patches = defaultdict(list)
        for r in range(nrows):
            for c in range(ncols):
                patch = image.crop((c * self.size, r * self.size,
                                    (c + 1) * self.size, (r + 1) * self.size))
                patches[r].append(patch)
        return patches

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        prefix = os.path.splitext(os.path.basename(image_path))[0]
        prefix = prefix.replace(self.suffix, '')

        patches = self.crop_to_patches(image)
        for row, row_patches in patches.items():
            for col, patch in enumerate(row_patches):
                patch.save(
                    os.path.join(self.opath, f"{prefix}_{row}_{col}.png"))

    def process_images(self):
        images = [
            os.path.join(self.ipath, file) for file in os.listdir(self.ipath)
            if file.lower().endswith('.png')
        ]

        images = sorted(images)
        if self.num_samples:
            images = images[:self.num_samples]

        with Pool() as p:
            p.map(self.preprocess_image, images)

    def __call__(self):
        if os.path.isdir(self.ipath) and os.path.isdir(self.opath):
            self.process_images()
        elif os.path.isfile(self.ipath) and os.path.isdir(
                self.opath) and self.ipath.lower().endswith('.png'):
            self.preprocess_image(self.ipath)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_samples', '-n', type=int, default=None)
    arg_parser.add_argument('--clean', '-c', action='store_true')
    arg_parser.add_argument('paths', nargs=argparse.REMAINDER)
    args = arg_parser.parse_args()

    num_samples = args.num_samples
    paths = args.paths
    clean = args.clean

    if len(paths) < 3 or len(paths) % 3:
        print(
            """Usage: python3 to_patch.py -n <num_samples> -c <input_path1> <output_dir1> <size1> <input_path2> <output_dir2> <size2> ...
                      -n: Number of samples to preprocess from each directory
                      -c: Clean the output directory before preprocessing
Example 1: python3 to_patch.py -n 200 -c \\
        ./data/Flickr2K/Flickr2K_HR ./data/minibatch/HR 100 \\
        ./data/Flickr2K/Flickr2K_LR_bicubic/X2 ./data/minibatch/LR 50

Example 2: python3 to_patch.py \\
        ./data/Flickr2K/Flickr2K_LR_bicubic/X2/000001x2.png ./data/minibatch/test 50
""")
        sys.exit(1)

    print(
        f'Preprocessing {num_samples} samples each from the following directories:\n'
    )

    for i in range(0, len(paths), 3):
        ipath, opath, size = paths[i], paths[i + 1], paths[i + 2]
        size = int(size)
        print(f'Input path: {ipath}')
        print(f'Output directory: {opath}')
        print(f'Patch size: {size}\n')

        if clean and os.path.exists(opath):
            for file in os.listdir(opath):
                os.remove(os.path.join(opath, file))

        ImageProcessor(ipath, opath, size, num_samples)()
