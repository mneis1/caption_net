import pickle
import utils.encoding as encoding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import keras.layers as layers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class rec_network:
    def __init__(self, model_name, dropout_rate=0.5, output_units=256, hidden_activation='relu', optimizer='adam'):
        self.name = model_name
        self.desc_list = pickle.load(open('descriptions.pkl', 'rb'))
        self.tokenizer = Tokenizer()
        self.get_tokenizer()
        self.max_len = self.get_max_len()
        self.word_count = self.count_words()
        
        if model_name == 'merge':
            self.model = self.merge_model(dropout_rate, output_units, hidden_activation, optimizer)
        elif model_name == 'inject':
            self.model = self.inject_model(dropout_rate, output_units, hidden_activation, optimizer)

        
    
    def get_model(self, model_name):
        if model_name == 'merge':
            return self.merge_model()
        elif model_name == 'inject':
            return self.inject_model()

    
    def merge_model(self, dropout_rate=0.5, output_units=256, hidden_activation='relu', optimizer='adam'):
        feats_input = layers.Input(shape=(4096,))
        feats_layer_1 = layers.Dropout(dropout_rate)(feats_input)
        feats_layer_2 = layers.Dense(output_units, activation=hidden_activation)(feats_layer_1)

        desc_input = layers.Input(shape=(self.max_len,))
        desc_layer_1 = layers.Embedding(self.word_count, output_units, mask_zero=True)(desc_input)
        desc_layer_2 = layers.Dropout(dropout_rate)(desc_layer_1)
        desc_layer_3 = layers.LSTM(output_units)(desc_layer_2)

        dec_layer_1 = layers.merge.add([feats_layer_2, desc_layer_3])
        dec_layer_2 = layers.Dense(output_units, activation=hidden_activation)(dec_layer_1)

        out_layer = layers.Dense(self.word_count, activation='softmax')(dec_layer_2)
        model = Model(inputs=[feats_input, desc_input], outputs=out_layer)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model


    def inject_model(self, dropout_rate=0.5, output_units=256, hidden_activation='relu', optimizer='adam'):
        feats_input = layers.Input(shape=(4096,))
        feats_layer_1 = layers.Dropout(dropout_rate)(feats_input)
        feats_layer_2 = layers.Dense(output_units, activation=hidden_activation)(feats_layer_1)

        desc_input = layers.Input(shape=(self.max_len,))
        desc_layer_1 = layers.Embedding(self.word_count, output_units, mask_zero=True)(desc_input)

        dec_layer_1 = layers.merge.add([feats_layer_2, desc_layer_1])
        dec_layer_2 = layers.Dropout(dropout_rate)(dec_layer_1)
        dec_layer_3 = layers.LSTM(output_units)(dec_layer_2)

        out_layer = layers.Dense(self.word_count, activation='softmax')(dec_layer_3)
        model = Model(inputs=[feats_input, desc_input], outputs=out_layer)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model


    def get_max_len(self):
        temp_list = []

        for k in self.desc_list.keys():
            for item in self.desc_list[k]:
                temp_list.append(item)

        length = max(len(item.split()) for item in temp_list)
        return length


    def train_model(self, epochs):
        
        feat_list = pickle.load(open('data_features.pkl', 'rb'))

        s = len(self.desc_list)

        for count in range(epochs):
            gen = self.compile_seq(feat_list)
            self.model.fit_generator(gen, epochs=1, steps_per_epoch=s, verbose=1)
            self.model.save('temp/rnn_models/'+str(self.name)+'_model_'+str(count)+'.h5')


    
    def get_tokenizer(self):
        string_list = []

        for k in self.desc_list.keys():
            for item in self.desc_list[k]:
                string_list.append(item)

        self.tokenizer.fit_on_texts(string_list)

    
    def encode_set(self, img, sub_desc_list):
        params_1 = []
        params_2 = []
        out_vals = []

        for item in sub_desc_list:

            set_ = self.tokenizer.texts_to_sequences([item])[0]
            set_len = len(set_)

            for val in range(1, set_len):
                set_in = set_[:val]
                set_out = set_[val]

                set_in = pad_sequences([set_in], maxlen=self.max_len)
                params_2.append(set_in[0])

                set_out = to_categorical([set_out], num_classes=self.word_count)
                out_vals.append(set_out[0])

                params_1.append(img)

        params_1 = np.asarray(params_1)
        params_2 = np.asarray(params_2)
        out_vals = np.asarray(out_vals)

        return params_1, params_2, out_vals
            


    def compile_seq(self, img_list):
        while True:
            for id_, desc in self.desc_list.items():
                in_1, in_2, out = self.encode_set(img_list[id_][0], desc)
                in_out_set = [[in_1,in_2],out]
                yield in_out_set

    def count_words(self):
        return len(self.tokenizer.word_index)+1
        