import os, sys
from multiprocessing import Pool
from PIL import Image


class PreprocessData:

    def __init__(self, data_directory, num_samples=None):
        self.NUM_SAMPLES = num_samples
        self.data_directory = data_directory
        self.hr_directory = os.path.join(data_directory,
                                         'Flickr2K/Flickr2K_HR')
        self.lr_directory = os.path.join(data_directory,
                                         'Flickr2K/Flickr2K_LR_bicubic/X2')

        self.hr_output_directory = os.path.join(data_directory, 'minibatch/HR')
        self.lr_output_directory = os.path.join(data_directory, 'minibatch/LR')
        self.hr_patch_size = (100, 100)
        self.lr_patch_size = (50, 50)
        if not os.path.exists(self.hr_output_directory):
            os.makedirs(self.hr_output_directory)
        if not os.path.exists(self.lr_output_directory):
            os.makedirs(self.lr_output_directory)

    def load_image(self, file_path):
        return Image.open(file_path)

    def crop_to_patches(self, image, patch_size):
        img_width, img_height = image.size
        width = (img_width // patch_size[0]) * patch_size[0]
        height = (img_height // patch_size[1]) * patch_size[1]
        image = image.crop((0, 0, width, height))

        patches = []
        for i in range(0, height, patch_size[1]):
            row = i // patch_size[1]
            for j in range(0, width, patch_size[0]):
                col = j // patch_size[0]
                patch = image.crop((j, i, j + patch_size[0], i + patch_size[1]))
                patches.append((patch, row, col))
        return patches

    def save_patches(self, patches, output_directory, file_prefix):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for patch, row, col in patches:
            patch.save(os.path.join(output_directory, f"{file_prefix}_{row}_{col}.png"))

    def preprocess_image(self, image_path, patch_size, save_dir):
        image = self.load_image(image_path)
        patches = self.crop_to_patches(image, patch_size)
        file_prefix = os.path.splitext(os.path.basename(image_path))[0]
        self.save_patches(patches, save_dir, file_prefix)

    def preprocess_hr_image(self, image_path):
        self.preprocess_image(image_path, self.hr_patch_size,
                              self.hr_output_directory)

    def preprocess_lr_image(self, image_path):
        self.preprocess_image(image_path, self.lr_patch_size,
                              self.lr_output_directory)

    def process_images(self):
        hr_images = [
            file for file in os.listdir(self.hr_directory)
            if file.endswith('.png')
        ]
        if self.NUM_SAMPLES:
            hr_images = hr_images[:self.NUM_SAMPLES]

        def hr_to_lr(hr_image):
            hr_image_name = os.path.splitext(hr_image)[0]
            return os.path.join(self.lr_directory, f"{hr_image_name}x2.png")

        lr_images = list(map(hr_to_lr, hr_images))
        hr_images = [
            os.path.join(self.hr_directory, file) for file in hr_images
        ]

        with Pool() as p:
            p.map(self.preprocess_hr_image, hr_images)
        print('Finished processing HR images')
        with Pool() as p:
            p.map(self.preprocess_lr_image, lr_images)
        print('Finished processing LR images')


if __name__ == "__main__":
    if len(sys.argv) == 2:
        num_samples = int(sys.argv[1])
    else:
        num_samples = None
    data_processor = PreprocessData('./data', num_samples)
    data_processor.process_images()
