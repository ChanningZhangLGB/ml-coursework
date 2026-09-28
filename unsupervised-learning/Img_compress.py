import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans

def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            images.append((filename, Image.open(img_path).convert('RGB')))
    return images

def compress_image(image, k):
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    new_pixels = np.array([kmeans.centers[label] for label in kmeans.predict(pixels)])
    new_image_np = new_pixels.reshape(image_np.shape)
    return Image.fromarray(np.uint8(new_image_np))

def visualize_and_save(images, ks=[3, 5, 10], output_dir='compressed_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename, image in images:
        plt.figure(figsize=(15, 8))
        plt.subplot(1, len(ks) + 1, 1)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')

        for i, k in enumerate(ks, start=2):
            compressed_image = compress_image(image, k)
            plt.subplot(1, len(ks) + 1, i)
            plt.imshow(compressed_image)
            plt.title(f'k={k}')
            plt.axis('off')
            
            compressed_image.save(os.path.join(output_dir, f'{filename[:-4]}_k{k}.png'))

        plt.tight_layout()
        plt.show()

# Path to your directory of images
image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_data')
images = load_images(image_dir)
visualize_and_save(images)