
# importing required libraries 
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input,decode_predictions
import numpy as np


def predict(model, img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) 
   
    # predicting the output 
    preds = model.predict(x)
    result_list = decode_predictions(preds, top=1)[0]
    result = tuple(result_list)
    result = result[0]
    result = result[1]
    return result 	

def load_model():
    # importing a pre-trained inception_v3 model 
    model = InceptionV3(weights='imagenet', include_top=True)
    return model

if __name__ == "__main__":
   
   # loading the model
   model = load_model()

   image_loc = "./images/"
   image_name = "1.jpg"
   image_path = image_loc + image_name 

   # output_predction
   result = predict(model, image_path)
   print (result)




