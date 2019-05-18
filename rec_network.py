import pickle
import utils.encoding as encoding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import keras.layers as layers


class rec_network:
    def __init__(self, model_name):
        self.name = model_name
        self.desc_list = pickle.load(open('descriptions.pkl', 'rb'))
        self.model = self.get_model(model_name)
        self.tokenizer = self.get_tokenizer()
        self.max_len = self.get_max_len()
        self.word_count = len(tokenizer.word_index)+1
    
    def get_model(self, model_name):
        if model_name == 'merge':
            return merge_model()

    
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
            gen = encoding.compile_seq(self.desc_list, feat_list, self.max_len, self.tokenizer)
            self.model.fit_generator(gen, epochs=epochs, steps_per_epoch=s, verbose=1)
            self.model.save('temp/rnn_models/'+str(self.name)+'_model_'+str(count)+'.h5')


    
    def get_tokenizer(self):
        string_list = []
        tk = Tokenizer()

        for k in self.desc_list.keys():
            for item in self.desc_list[k]:
                string_list.append(item)

        tk.fit_on_texts(string_list)
        return tk