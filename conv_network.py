from keras.applications import vgg16, inception_v3
from keras.models import Model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import pickle


class conv_network:
    def __init__(self, imgnet_model_name):
        self.model_name = imgnet_model_name
        self.imgnet_model, self.model_params = self.fetch_model(imgnet_model_name) #load_model
        #remove softmax layer
        self.model = Model(inputs=self.imgnet_model.input, outputs=self.imgnet_model.layers[-2].output)
        #self.model_params = fetch_model_params(imgnet_model_name)
        

    def get_img_features(self, image_path):
        dim = self.model_params['resize_dims']
        img = image.load_img(image_path, target_size=(dim,dim))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        #img = img.reshape((-1,img.shape[0],img.shape[1],img.shape[2]))
        img = self.img_preprocessing(img)
        pred = self.model.predict(img, verbose=0)

        return pred

    def img_preprocessing(self, img):
        model_name_ = self.model_name

        if model_name_ == 'vgg16':
            return vgg16.preprocess_input(img)
        elif model_name_ == 'inception_v3':
            return inception_v3.preprocess_input(img)

    def compile_features(self, data_path):
        features_dict = dict()
        for image in os.listdir(data_path):
            image_path = data_path + '/' + image
            img_id, _ = os.path.splitext(image)
            features_dict[img_id] = self.get_img_features(image_path)
        
        #dump features to file so program doesnt have to compute everytime
        dump_loc = open('data_features.pkl','wb')
        pickle.dump(features_dict, dump_loc)
            


    def fetch_model(self, model_name):
        if model_name == 'vgg16':
            model = vgg16.VGG16()
            model_params = {'resize_dims': 224}
        elif model_name == 'inception_v3':
            model = inception_v3.InceptionV3()
            model_params = {'resize_dims': 299}
        #TODO: add error for incorrect model name
        return model, model_params

    def fetch_model_params(self, model_name):
        #get expected resize model expects and other params specific to the pretrained net
        if model_name == 'vgg16':
            model_params = {'resize_dims': 224}
        elif model_name == 'inception_v3':
            model_params = {'resize_dims': 299}

        return model_params

    def load_features(self, dir='data_features.pkl'):
        return pickle.load(open(dir, 'rb'))

    #returns features that are in a given set of data. data should be a list of data id's to compare
    def filter_features(self, features, data):
        return {id_: features[id_] for id_ in data}