
from PIL import Image
import numpy as np

path = "/lhome/risverm/workspace/V3H/Doc_to_Daimler/Daimler/Cityscapes/input_v3h/img/img_1.ppm"
im = Image.open(path)
im.save("test0.ppm")
n_im = np.array(im, dtype=np.uint8)
p_im = np.divide(n_im, 255)
p_im = np.multiply(p_im, 255)
print(p_im.shape)
save_im = np.array(p_im, dtype=np.uint8)
im_arr = Image.fromarray(save_im).save("test.ppm")
new_image = Image.open("test.ppm")
n_im = np.array(new_image, dtype=np.uint8)
p_im = np.multiply(n_im, [255, 255, 255])
save_im = np.array(p_im, dtype=np.uint8)
new_im_arr = Image.fromarray(save_im).save("test1.ppm")
