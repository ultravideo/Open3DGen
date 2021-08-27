import numpy as np
import cv2
import glob
from copy import deepcopy

frames = glob.glob("out/source_images/*.png")
frames = sorted(frames)
frames = reversed(frames)


stacked = None

def nonzero_to_mask(s):
    mask = deepcopy(s)

    mask[mask != 0] = 2
    mask[mask == 0] = 1
    mask[mask == 2] = 0

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    
    return mask

i = 0
for f in frames:
    print(f)
    img = cv2.imread(f)
    showimg = cv2.resize(img, (1024, 1024))

    if stacked is None:
        stacked = img

    else:
        mask = nonzero_to_mask(img)
        stacked = stacked * mask + img

        # stacked += img

    i += 1
    print("frame:", i)

    
    stacked_downres = cv2.resize(stacked, (1024, 1024))


    cv2.imshow("img", np.hstack((showimg, stacked_downres)))

    key = cv2.waitKey(1)
    if key == ord('q'):
        quit()
    elif key == ord('r'):
        print("saved to", "out/combined_texture_" + str(i) + ".png")
        cv2.imwrite("out/combined_texture_" + str(i) + ".png", stacked)
        quit()

print("saved to out/combined_texture_full.png")
cv2.imwrite("out/combined_texture_full.png", stacked)