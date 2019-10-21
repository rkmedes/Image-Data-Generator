import os
import cv2
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import numpy as np

def generate_images(folder_from,folder_to,num_of_images):
    for filename in listdir(folder_from):
        path=os.path.join(folder_from,filename)
        datagen=ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,
                                   zoom_range=0.2,horizontal_flip=True)
        image=cv2.imread(path)
        x=img_to_array(image)/255.
        x=x.reshape((1,)+x.shape)
        count=1
        file=filename.split('.')
        folder_in=os.path.join(folder_to,file[0])
        try:
            if os.path.exists(folder_in): # creating seperate folders for training images
                os.remove(folder_in)
            os.makedirs(folder_in)
        except OSError:
            pass
        for batch in datagen.flow(x,batch_size=1):
            image_name=file[0]+str(count)+"."+file[1]
            cv2.imwrite(folder_in+'/'+image_name,image)
            if count==num_of_images:
                break
            count+=1
