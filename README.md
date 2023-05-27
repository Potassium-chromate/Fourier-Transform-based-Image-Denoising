# Fourier Transform based Image Denoising
This repository contains a MATLAB script for performing image denoising using the Discrete Fourier Transform (DFT).

The algorithm works by first adding noise to an image, then applying a Fourier Transform to transform the image from the spatial domain to the frequency domain. Then, a filter is applied to the transformed image to remove the noise. The filtered image is then transformed back to the spatial domain using the Inverse Discrete Fourier Transform (IDFT). The algorithm can handle both grayscale and RGB images.

## Overview
- Load an image and add artificial noise.
- Apply the DFT to the noisy image and shift the frequency components to center the Fourier Transform image.
- Create a rectangular filter that will isolate the main frequency components of the image.
- Apply this filter to the Fourier Transform of the noisy image.
- Perform an inverse shift and IDFT to transform the filtered image back to the spatial domain.
## Contents
The MATLAB script includes functions for performing the DFT and IDFT, as well as Fourier shifting and inverse shifting, and normalization. Additionally, it includes a function for adding random noise points to an image.

The main steps of the script are visualized with plots of the original image, the noisy image, the Fourier Transforms of the original and noisy images, and the filtered image after denoising.

## Usage
1. Load an image into MATLAB.
2. Choose a radius for the filter.
3. Run the script.
Please note that you must ensure the image file ('lena.jpg' in the example) is in your MATLAB's current directory or specify the correct path to the image file.

## Dependencies
This code is written in MATLAB and does not require any additional toolboxes.

## Disclaimer
This is a simple demonstration of image denoising using the Fourier Transform. The quality of denoising might not be perfect, especially for real-world noisy images. For more advanced noise reduction techniques, consider using methods such as wavelet transform or deep learning-based approaches.




