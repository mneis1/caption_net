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

test_conv = conv_network('VGG16')

train_ids = t_utils.get_file_list('data/Flickr8k_text/Flickr_8k.trainImages.txt')

train_desc = t_utils.filter_desc_list(train_ids)
train_feats = test_conv.filter_features(test_conv.load_features(), train_ids)

#print(list(train_desc)[0])


test_rec = rec_network('merge', train_feats, train_desc)

#%%
test_rec.train_model(5)

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