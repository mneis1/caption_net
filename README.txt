EXAMPLE ON HOW TO RUN:

Unpack Flickr 8k to data directory so text and images are in their own folders. Can use Flickr 30k or MSCOCO but hardware intensive

python main.py train --img_data_path data/set --img_names img/names --descriptions_path some/token/txt --conv_network vgg16_or_inception --model merge_or_inject --train_epochs epochs

python main.py predict --img_data_path some/img.jpg --conv_network vgg16_or_inception --model_path path/to/model.h5

python main.py eval --img_names img/names --conv_network vgg16_or_inception --model_path path/to/model.h5

python main.py eval_all --img_names img/names --conv_network vgg16_or_inception --model_path path/to/model_dir

