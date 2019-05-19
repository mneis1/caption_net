#%%

from conv_network import conv_network

#%%

test_net = conv_network('VGG16')
test_feats = test_net.compile_features('./data/Flicker8k_Dataset')
print(test_feats)
#%%
import utils.text_utils as t_utils

train_ids = t_utils.get_file_list('data/Flickr8k_text/Flickr_8k.trainImages.txt')
desc_test = t_utils.compile_description_list('data/Flickr8k_text/Flickr8k.token.txt', train_ids)

#%%
from rec_network import rec_network
import utils.encoding as encode
import pickle

test_rec = rec_network('merge')

#desc_train = pickle.load(open('descriptions.pkl', 'rb'))
feat_train = pickle.load(open('data_features.pkl', 'rb'))
print(test_rec.word_count)

#%%
generator = test_rec.compile_seq(img_list=feat_train)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)