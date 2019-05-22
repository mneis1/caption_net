#%%

from conv_network import conv_network

#%%

#test_net = conv_network('VGG16')
#test_feats = test_net.compile_features('./data/Flicker8k_Dataset')
#print(test_feats)
#%%
import utils.text_utils as t_utils

train_ids = t_utils.get_file_list('data/Flickr8k_text/Flickr_8k.trainImages.txt')
desc_test = t_utils.compile_description_list('data/Flickr8k_text/Flickr8k.token.txt')
train_descs = t_utils.filter_desc_list(train_ids)

#%%
from rec_network import rec_network
import utils.encoding as encode
import pickle
from conv_network import conv_network
import utils.text_utils as t_utils

#Just train

test_conv = conv_network('vgg16')

train_ids = t_utils.get_file_list('data/Flickr8k_text/Flickr_8k.trainImages.txt')

train_desc = t_utils.filter_desc_list(train_ids)
train_feats = test_conv.filter_features(test_conv.load_features(), train_ids)

#print(list(train_desc)[0])


train_merge = rec_network('merge', train_feats, train_desc)
train_inject = rec_network('inject', train_feats, train_desc)


#%%
train_merge.train_model(3)
train_inject.train_model(3)

#%%
#test network

import utils.eval_suite as eval_s

test_ids = t_utils.get_file_list('data/Flickr8k_text/Flickr_8k.testImages.txt')

test_desc = t_utils.filter_desc_list(test_ids)
test_feats = test_conv.filter_features(test_conv.load_features(), test_ids)

merge_bl4_scores = eval_s.evaluate_BLEU4(test_feats, test_desc, 'merge', 'temp/rnn_models/merge_model_0.h5', 34)
inject_bl4_scores = eval_s.evaluate_BLEU4(test_feats, test_desc, 'inject', 'temp/rnn_models/inject_model_0.h5', 34)


print(merge_bl4_scores)
print(inject_bl4_scores)

#%%
#Predict example
from rec_network import rec_network
from conv_network import conv_network
from PIL import Image


pred_net = rec_network('merge', [], [], 'temp/rnn_models/merge_model_0.h5', 'tokenizer.pkl', 34)
feat_net = conv_network('vgg16')
feats = feat_net.get_img_features('data/Flicker8k_Dataset/3737492755_bcfb800ed1.jpg')
pred_cap = pred_net.predict_caption(feats)

print(pred_cap)

img = Image.open('data/Flicker8k_Dataset/3737492755_bcfb800ed1.jpg')
img


#%%
#desc_train = pickle.load(open('descriptions.pkl', 'rb'))
feat_train = pickle.load(open('data_features.pkl', 'rb'))
print(test_rec.tokenizer.word_index)

#%%
generator = test_rec.compile_seq(img_list=feat_train)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)

#%%
import utils.eval_suite as eval
import utils.text_utils as t_utils

file = t_utils.get_file_list('data/Flickr8k_text/Flickr_8k.testImages.txt')
desc_t = t_utils.compile_description_list()

bleu_test = eval.evaluate_BLEU4(test_img_set, test_desc_set, model_name, model_path)


#%%
test_rec.train_model(10)