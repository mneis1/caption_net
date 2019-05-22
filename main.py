import sys
import argparse
from conv_network import conv_network
import utils.text_utils as t_utils
from rec_network import rec_network
import utils.eval_suite as eval_
import os

RNN_MODELS = ['merge', 'inject']
CNN_MODELS = ['vgg16', 'inception_v3']
FUNC_OPTIONS = ['train', 'predict', 'eval', 'eval_all']


def main(args):
    #need to have a train and eval option. if arg is train it does one thing, if eval, another
    parser = argparse.ArgumentParser()

    parser.add_argument('function', choices=FUNC_OPTIONS)
    parser.add_argument('--conv_network', choices=CNN_MODELS)
    parser.add_argument('--model', choices=RNN_MODELS)
    parser.add_argument('--model_path', type=str, help='dont enter trailing / when entering in a directoy')
    parser.add_argument('--img_data_path', type=str, help='when training/evaluating, pass path to dir of imgs. \nwhen prediciting send path to single img.')
    parser.add_argument('--img_names', type=str)
    parser.add_argument('--descriptions_path', type=str)
    parser.add_argument('--train_epochs')


    args_ = parser.parse_args(args)

    print("Working...")

    if args_.function == 'train':
        #python main.py train --img_data_path data/set --img_names img/names --descriptions_path some/token/txt --conv_network vgg16_or_inception --model merge_or_inject --train_epochs epochs

        conv = conv_network(args_.conv_network)
        conv.compile_features(args_.img_data_path)
        t_utils.compile_description_list(args_.descriptions_path)

        train_names = t_utils.get_file_list(args_.img_names)
        train_descs = t_utils.filter_desc_list(train_names)
        train_feats = conv.filter_features(conv.load_features(), train_names)

        rec = rec_network(args_.model, train_feats, train_descs)

        rec.train_model(args_.train_epochs)


    elif args_.function == 'predict':
        #python main.py predict --img_data_path some/img.jpg --conv_network vgg16_or_inception --model_path path/to/model.h5


        conv = conv_network(args_.conv_network)
        rec = rec_network(args_.model_path, [], [], args_.model_path, 'tokenizer.pkl', 34)
        feats = conv.get_img_features(args_.img_data_path)
        cap = rec.predict_caption(feats)

        split_cap = cap.split(' ')
        split_cap.remove('startcap')
        split_cap.remove('stopcap')
        new_cap = ' '.join(split_cap)

        print('Caption for image ' + args_.img_data_path + ':')
        print('\"' + new_cap + '.\"')
        

    elif args_.function == 'eval':
        #python main.py eval --img_names img/names --conv_network vgg16_or_inception --model_path path/to/model.h5


        conv = conv_network(args_.conv_network)

        test_ids = t_utils.get_file_list(args_.img_names)
        test_desc = t_utils.filter_desc_list(test_ids)
        test_feats = conv.filter_features(conv.load_features(), test_ids)

        bl4 = eval_.evaluate_BLEU4(test_feats, test_desc, args_.model_path, args_.model_path, 34)

        print('BLEU-4 Scores for model ' + '\"' + args_.model_path + '\"')
        print("(The closer to 1 BLEU-n is, the more accurate a description is to the test example.)")
        print('\"')
        print('BLEU-1: ' + bl4[0])
        print('BLEU-2: ' + bl4[1])
        print('BLEU-3: ' + bl4[2])
        print('BLEU-4: ' + bl4[3])
        

    elif args_.function == 'eval_all':

        #python main.py eval_all --img_names img/names --conv_network vgg16_or_inception --model_path path/to/model_dir

        models = os.listdir(args_.model_path)
        conv = conv_network(args_.conv_network)
        test_ids = t_utils.get_file_list(args_.img_names)
        test_desc = t_utils.filter_desc_list(test_ids)
        test_feats = conv.filter_features(conv.load_features(), test_ids)

        for model in models:
            
            model_path_ = args_.model_path +'/' + model

            bl4 = eval_.evaluate_BLEU4(test_feats, test_desc, model_path_, model_path_, 34)

            print('BLEU-4 Scores for model ' + '\"' + model + '\"')
            print("(The closer to 1 BLEU-n is, the more accurate a description is to the test example.)")
            print('\n')
            print('BLEU-1: ' + str(bl4[0]))
            print('BLEU-2: ' + str(bl4[1]))
            print('BLEU-3: ' + str(bl4[2]))
            print('BLEU-4: ' + str(bl4[3]))
            print('----------------------------------------------------------------------')




    

    
    return

if __name__ == '__main__':
    main(sys.argv[1:])