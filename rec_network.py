import pickle
import utils.encoding as encoding
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
import keras.layers as layers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np


#TODO: need a way to save an reload params such as max length

class rec_network:
    def __init__(self, model_name, train_feature_list, train_description_list, model_path=None, tokenizer_path=None, max_length=None, dropout_rate=0.5, output_units=256, hidden_activation='relu', optimizer='adam'):
        self.name = model_name
        self.desc_list = train_description_list
        self.feat_list = train_feature_list
        if tokenizer_path == None:
            self.tokenizer = Tokenizer()
            self.get_tokenizer()
        else:
            self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))

        self.max_len = self.get_max_len() if max_length is None else max_length
        self.word_count = self.count_words()

        if model_path == None:
            if model_name == 'merge':
                self.model = self.merge_model(dropout_rate, output_units, hidden_activation, optimizer)
            elif model_name == 'inject':
                self.model = self.inject_model(dropout_rate, output_units, hidden_activation, optimizer)
        else:
            self.model = load_model(model_path)


    
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
            [temp_list.append(i) for i in self.desc_list[k]]
        length = max(len(i.split()) for i in temp_list)
        return length


    def train_model(self, epochs):
        
        feat_list = self.feat_list

        s = len(self.desc_list)

        for count in range(epochs):
            self.shuffle_feats()
            gen = self.compile_seq()
            self.model.fit_generator(gen, epochs=1, steps_per_epoch=s, verbose=1)
            self.model.save('temp/rnn_models/'+str(self.name)+'_model_'+str(count)+'.h5')
            


            

    def predict_caption(self, img):
        cap = ['startcap']
        for count in range(self.max_len):
            next_word = None
            prev_set = ' '.join(cap)
            set_ = self.tokenizer.texts_to_sequences([prev_set])[0]
            set_ = pad_sequences([set_], maxlen=self.max_len)
            next_ = np.argmax(self.model.predict([img,set_], verbose=0))

            for id_, ind in self.tokenizer.word_index.items():
                if ind == next_:
                    next_word = id_

            if next_word == None:
                break
            else:
                cap.append(next_word)

                if next_word == 'stopcap':
                    break

        return ' '.join(cap)            



    def get_tokenizer(self):
        string_list = []

        for k in self.desc_list.keys():
            for item in self.desc_list[k]:
                string_list.append(item)

        self.tokenizer.fit_on_texts(string_list)
        pickle.dump(self.tokenizer, open('tokenizer.pkl','wb'))

    
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
            


    def compile_seq(self):
        while True:
            for id_, desc in self.desc_list.items():
                in_1, in_2, out = self.encode_set(self.feat_list[id_][0], desc)
                in_out_set = [[in_1,in_2],out]
                yield in_out_set

    def count_words(self):
        return len(self.tokenizer.word_index)+1

    def shuffle_feats(self):
        keys = list(self.feat_list.keys())
        np.random.shuffle(keys)
        shuffled_dict = {k: self.feat_list[k] for k in keys}
        self.feat_list = shuffled_dict

        