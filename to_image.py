from PIL import Image
from collections import defaultdict
import os, sys
from multiprocessing import Pool


class ImageStitcher:

    def __init__(self, ipath, opath):
        self.ipath = ipath
        self.opath = opath
        self.images = defaultdict(list)

        if not os.path.exists(self.opath):
            os.makedirs(self.opath)

    def stitch_image_from_prefix(self, patch_files):
        max_row = max(patch_files, key=lambda x: x['row'])['row']
        max_col = max(patch_files, key=lambda x: x['col'])['col']
        prefix = patch_files[0]['prefix']

        size = Image.open(patch_files[0]['path']).size[0]
        total_width = size * (max_col + 1)
        total_height = size * (max_row + 1)
        full_image = Image.new('RGB', (total_width, total_height))

        for file in patch_files:
            image = Image.open(file['path'])
            full_image.paste(image, (file['col'] * size, file['row'] * size))

        full_image.save(os.path.join(self.opath, f"{prefix}.png"))

    def __call__(self):
        for file in os.listdir(self.ipath):
            if not file.lower().endswith('.png'):
                continue
            attrs = file.split('_')
            if len(attrs) != 3:
                print(f"Skipping {file} due to incorrect naming format")
            prefix, row, col = os.path.splitext(file)[0].split('_')
            self.images[attrs[0]].append({
                'prefix': prefix,
                'row': int(row),
                'col': int(col),
                'path': os.path.join(self.ipath, file)
            })

        with Pool() as p:
            p.map(self.stitch_image_from_prefix, self.images.values())


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python to_image.py <input_dir> <output_dir>")
        sys.exit(1)

    ipath, opath = sys.argv[1], sys.argv[2]
    ImageStitcher(ipath, opath)()
