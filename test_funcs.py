#%%

from conv_network import conv_network

#%%

test_net = conv_network('VGG16')
test_feats = test_net.compile_features('./data/Flicker8k_Dataset')
test_feats

#%%
import utils.text_utils as t_utils

train_ids = t_utils.get_file_list('data/Flickr8k_text/Flickr_8k.trainImages.txt')
desc_test = t_utils.compile_description_list('data/Flickr8k_text/Flickr8k.token.txt', train_ids)