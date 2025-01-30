import os
import numpy as np
import shutil
from random import shuffle
from PIL import Image
import pydicom
import cv2


def resize_images(source_dir, dest_dir, size=(Image_H, Image_W)):
    """
    Resize all .jpg files in the source directory to the specified size and save them to the destination directory.
    This function resizes all JPEG and PNG files in a specified source directory to a given size and saves them in a destination directory. 
    It also converts JPEG images to PNG format during this process. The function creates the destination directory if it doesn't already exist, 
    reads each image, resizes it, and then saves it to the destination directory while printing a message to indicate the action.
    """
    # Create the destination directory if it does not exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
            # Define the source and destination file paths            
            if filename.lower().endswith('.jpg'):
                source_path = os.path.join(source_dir, filename)
                dest_path = os.path.join(dest_dir, filename.replace('.jpg', '.png'))
            else:
                source_path = os.path.join(source_dir, filename)
                dest_path = os.path.join(dest_dir, filename)
            # dest_path = os.path.join(dest_dir, filename.replace('.jpg', '.png'))
            # dest_path = os.path.join(dest_dir, filename)

            # Read the image from file
            image = cv2.imread(source_path)
            
            # Resize the image
            resized_image = cv2.resize(image, size)

            # Save the resized image to the destination path
            cv2.imwrite(dest_path, resized_image)
            print(f"Copied {source_path} to {dest_dir}")
            


'''
These functions perform advanced image processing to simulate dark-field microscopy effects:
    The first version applies a high-pass filter using a Laplacian kernel to emphasize edges in an image. 
    It then normalizes and inverts the image to simulate dark-field scattering effects. 
    Finally, it adjusts the image's gamma based on its mean brightness, which dynamically enhances or reduces the image contrast.

    The second version simplifies this process by using a fixed gamma value for gamma correction 
    after applying the high-pass filter and inverting the image.
'''

laplacian_kernel = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]], dtype=np.float32)


def image_preprocessing1(image_path, ddepth=-1, max=0.6, min=0.1, kernel= laplacian_kernel):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale
    if image is None:
        raise ValueError("The image could not be loaded. Check the file path.")
    
    # Apply high-pass filter to emphasize edges
    high_pass = cv2.filter2D(image, ddepth, kernel)
    
    high_pass = cv2.normalize(high_pass, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Invert the image to simulate dark-field's scattering emphasis
    inverted = cv2.bitwise_not(high_pass)
    
    # Automatically adjust gamma based on the image brightness
    mean_brightness = np.mean(inverted)
    gamma = np.interp(mean_brightness, [0, 255], [max, min])  # Calibration may be required

    # Apply gamma correction
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    simulated_darkfield = cv2.LUT(inverted, look_up_table)
    
    return simulated_darkfield


def image_preprocessing2(image_path, gamma=0.8, ddepth=-1, kernel=laplacian_kernel):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale
    
    high_pass = cv2.filter2D(image, ddepth, laplacian_kernel)
    
    inverted = cv2.bitwise_not(high_pass)    
    
    look_up_table = np.array([((i / 255.0) ** gamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            
    simulated_darkfield = cv2.LUT(inverted, look_up_table)
    
    return simulated_darkfield