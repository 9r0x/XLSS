from PIL import Image
import os
import re

def stitch_images_from_id(image_id, patch_folder, output_folder):
    image_load = re.compile(f"{image_id}(x2)?_(\d+)_(\d+).png")
    files = [f for f in os.listdir(patch_folder) if image_load.match(f)]

    max_row = 0
    max_col = 0
    for file in files:
        row, col = map(int, image_load.search(file).groups()[1:3]) 
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    
    first_image = Image.open(os.path.join(patch_folder, files[0]))
    width, height = first_image.size

    total_width = width * (max_col + 1)
    total_height = height * (max_row + 1)
    full_image = Image.new('RGB', (total_width, total_height))

    for file in files:
        row, col = map(int, image_load.search(file).groups()[1:3])
        image = Image.open(os.path.join(patch_folder, file))
        full_image.paste(image, (col * width, row * height))
    os.makedirs(output_folder, exist_ok=True)
    full_image.save(os.path.join(output_folder, f"{image_id}_stitch.png"))

if __name__ == "__main__":
    stitch_images_from_id("000387", "./data/minibatch/HR", "./data/stitch")
