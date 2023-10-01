from PIL import ImageEnhance
import os
import numpy as np
from PIL import Image

# Enhances the brightness of the image
def brightnessEnhancement(root_path,img_name):
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    # brightness = 1.1+0.4*np.random.random()
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

# Enhances the contrast of the image
def contrastEnhancement(root_path, img_name):
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.1+0.4*np.random.random()
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

# Performs a random rotation of the image by an angle of -90, 0, or 90 degrees
def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.randint(-2, 2)*90
    if random_angle==0:
     rotation_img = img.rotate(-90)
    else:
        rotation_img = img.rotate( random_angle)
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

# Horizontally flips the image
def flip(root_path,img_name):
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img


def createImage(imageDir,saveDir):
   i=0
   for name in os.listdir(imageDir):
      i=i+1
      saveName="cesun"+str(i)+".jpg"
      saveImage=contrastEnhancement(imageDir,name)
      saveImage.save(os.path.join(saveDir,saveName))
      saveName1 = "flip" + str(i) + ".jpg"
      saveImage1 = flip(imageDir,name)
      saveImage1.save(os.path.join(saveDir, saveName1))
      saveName2 = "brightnessE" + str(i) + ".jpg"
      saveImage2 = brightnessEnhancement(imageDir, name)
      saveImage2.save(os.path.join(saveDir, saveName2))
      saveName3 = "rotate" + str(i) + ".jpg"
      saveImage = rotation(imageDir, name)
      saveImage.save(os.path.join(saveDir, saveName3))


if __name__ == '__main__':
    parser.add_argument('-img_dir',type=str, help='input image directory')
    parser.add_argument('-increimg_dir', type=str, help='output image directory')

    args = parser.parse_args()