from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer

def encode_set(desc_list, img, max_len, tokenizer, no_of_words):
    params_1 = []
    params_2 = []
    out_vals = []

    for item in desc_list:
        set_ = tokenizer.texts_to_sequences([item])
        set_ = set_[0]
        set_len = len(set_)

        for val in range(1, set_len):
            set_in = set_[:val]
            set_out = set_[val]

            set_in = pad_sequences([set_in], maxlen=max_len)
            params_2.append(set_in[0])

            set_out = to_categorical([set_out], num_classes=no_of_words)
            out_vals.append(set_out[0])

            params_1.append(img)

    params_1 = np.asarray(params_1)
    params_2 = np.asarray(params_2)
    out_vals = np.asarray(out_vals)

    return params_1, params_2, out_vals
            


def compile_seq(desc_list, img_list, max_len, tokenizer, word_count):
    while True:
        for id_, desc in desc_list.items():
            print(type(tokenizer))
            in_1, in_2, out = encode_set(desc, img_list[id_][0], max_len, tokenizer, word_count)
            in_out_set = [[in_1,in_2],out]
            yield in_out_set