from PIL import Image
import os


basewidth = 640
photo_path = 'dataset_640x480/Sekcje'


for section in os.listdir(photo_path):
    section_path = os.path.join(photo_path, section)
    for photo in os.listdir(section_path):
        path = os.path.join(photo_path, section, photo)

        img = Image.open(path)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(path)


