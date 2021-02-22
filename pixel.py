#!/usr/bin/env python3

import numpy as np
import cv2

def QuantizeToGivenPalette(im, palette):
    """Quantize image to a given palette.

    The input image is expected to be a Numpy array.
    The palette is expected to be a list of R,G,B values."""

    # Calculate the distance to each palette entry from each pixel
    distance = np.linalg.norm(im[:,:,None] - palette[None,None,:], axis=3)

    # Now choose whichever one of the palette colours is nearest for each pixel
    palettised = np.argmin(distance, axis=2).astype(np.uint8)

    return palettised

# Open input image and palettise to "inPalette" so each pixel is replaced by palette index
# ... so all black pixels become 0, all red pixels become 1, all green pixels become 2...
im=cv2.imread("test13.png",cv2.IMREAD_COLOR)

inPalette = np.array([
   [0,0,0],             # black
   [0,0,255],           # red
   [0,255,0],           # green
   [255,0,0],           # blue
   [255,255,255]],      # white
   dtype=np.uint8)

r = QuantizeToGivenPalette(im,inPalette)

# Now make LUT (Look Up Table) with the 5 new colours
LUT = np.zeros((5,3),dtype=np.uint8)
LUT[0]=[255,255,255]  # white
LUT[1]=[255,255,0]    # cyan
LUT[2]=[255,0,255]    # magenta
LUT[3]=[0,255,255]    # yellow
LUT[4]=[0,0,0]        # black

# Look up each pixel in the LUT
result = LUT[r]

# Save result
cv2.imwrite('result.png', result)