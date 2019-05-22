import nltk
from rec_network import rec_network

def evaluate_BLEU4(test_img_set, test_desc_set, model_name, model_path, max_len, tokenizer_path='tokenizer.pkl'):
    eval_net = rec_network(model_name, test_img_set, test_desc_set, model_path, tokenizer_path, max_length=max_len)
    test_set = []
    pred_set = []

    b1_w = (1, 0, 0, 0)
    b2_w = (.5, .5, 0, 0)
    b3_w = (.33, .33, .33, 0)
    b4_w = (.25 ,.25 ,.25 ,.25)

    for id_, item in test_desc_set.items():
        pred = eval_net.predict_caption(test_img_set[id_])
        test_set.append([desc.split() for desc in item])
        pred_set.append(pred.split())

    bleu1 = nltk.translate.bleu_score.corpus_bleu(test_set, pred_set, weights=b1_w)
    bleu2 = nltk.translate.bleu_score.corpus_bleu(test_set, pred_set, weights=b2_w)
    bleu3 = nltk.translate.bleu_score.corpus_bleu(test_set, pred_set, weights=b3_w)
    bleu4 = nltk.translate.bleu_score.corpus_bleu(test_set, pred_set, weights=b4_w)

    return [bleu1, bleu2, bleu3, bleu4]




