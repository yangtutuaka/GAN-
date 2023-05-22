from PIL import Image
import os.path
import glob
def convertpng(pngfile,outdir,width=64,height=64):
    img=Image.open(pngfile)
    new_img=img.resize((width,height),Image.BILINEAR)
    new_img.save(os.path.join(outdir,os.path.basename(pngfile)))
for pngfile in glob.glob(r"F:\360Downloads\faces\*.jpg"):
    convertpng(pngfile,r"F:\360Downloads\faces_64")