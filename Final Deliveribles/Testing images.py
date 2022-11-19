from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

import numpy as np

model=load_model("Fitness.h5")

img=image.load_img(r"C:\Users\Azhagan\Desktop\Untitled Folder\test\pineee.jpeg",grayscale=False,target_size=(64,64))
                   
x=image.img_to_array(img)

x=np.expand_dims(x,axis=0)

pred=np.argmax(model.predict(x),axis=1)
print("predition",pred)



index=['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON']


result=str(pred[index[0]])

print(result)
