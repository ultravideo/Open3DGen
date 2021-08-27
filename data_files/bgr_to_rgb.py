import cv2

image = "data_files/out/texture_out_gpu_average_dilated.png"

img = cv2.imread(image)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(image, rgb)
