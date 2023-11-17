import data_engineering as de
from PIL import Image 





mirror_pic = de.makemirror("data/training/img/01_g.jpg",512,512,32)
mirror_tree = de.makemirror("data/training/tree/01_g.tif",512,512,32)
mirror_test = de.makemirror("data/training/img/02_g.jpg",512,512,32)

croped_pic = de.crop(mirror_pic ,512,512,32)
croped_tree = de.crop(mirror_tree ,512,512,32)
croped_test = de.crop(mirror_test ,512,512,32)

gray_scale_pic = de.grayscale_cropped_pieces(croped_pic)
gray_scale_test = de.grayscale_cropped_pieces(croped_test)



de.save_cropped_pieces(gray_scale_pic, "unet/data/membrane/train/image")
de.save_cropped_pieces(croped_tree, "unet/data/membrane/train/label")
de.save_cropped_pieces(gray_scale_test,"unet/data/membrane/test")


de.process_images("data/training/img","unet/data/membrane/train/image")
de.process_images("data/training/tree","unet/data/membrane/train/label")
#de.process_images("")