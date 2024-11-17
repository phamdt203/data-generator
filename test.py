import os
import rawpy
import cv2
import imageio

def convert_and_reduce_noise(raw_path, output_path, h=10):
    try:
        # Open the RAW image using rawpy
        with rawpy.imread(raw_path) as raw:
            # Convert to RGB image
            rgb_image = raw.postprocess()

            # Apply noise reduction using OpenCV
            # Parameters for noise reduction can be tuned
            denoised_image = cv2.fastNlMeansDenoisingColored(
                src=rgb_image,
                h=h,  # Filter strength for luminance component
                hForColorComponents=h,  # Filter strength for color components
                templateWindowSize=7,  # Size of the template patch
                searchWindowSize=21  # Size of the window used to search similar patches
            )

            # Save the denoised image as JPEG
            imageio.imwrite(output_path, denoised_image)
            print(f'Noise reduced and saved: {output_path}')

    except Exception as e:
        print(f'Error processing {raw_path}: {e}')

if __name__ == "__main__":
    next
    # Define the folder containing RAW files
    raw_folder = r'/media/ubuntu/3132-6638/DCIM/101CANON/'
    raw_file = 'IMG_7869.CR2'  # Specify your RAW file
    raw_path = os.path.join(raw_folder, raw_file)

    # Define output path
    output_path = os.path.join(raw_folder, 'JPEG', os.path.splitext(raw_file)[0] + '_denoised.jpg')

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert and reduce noise
    convert_and_reduce_noise(raw_path, output_path, h=10)
