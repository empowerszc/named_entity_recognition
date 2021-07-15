import sys
import argparse
import os
from collections import Counter

from utils.data_util import build_corpus,load_params
from utils.preprocess_util import save_model,load_model,flatten_lists,extend_maps,flatten_lists
from utils.evaluating import Metrics
from utils.config import MODEL_LIST
from models.model_factory import getModel

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Named Entity Recognition")
parser.add_argument("--model", type=str, default="BILSTM_CRF")
parser.add_argument("--dataset", type=str, default="ResumeNER")
parser.add_argument("--text", type=str, default="钱学森，中国著名科学家，中科院院士")
args = parser.parse_args()


def predict():
    model_name = args.model
    dataset = args.dataset
    data_dir = "./datasets/" + dataset + "/data/"
    model_path = "./datasets/"+dataset + "/ckpts/" + model_name.lower()+".pkl"

    if model_name == "BILSTM_CRF":
        for_crf = True
    else:
        for_crf = False

    text =  args.text
    word_list = list(text.replace(" ", ""))

    word_tag_map = load_params(data_dir)
    word2id = word_tag_map["word2id"]
    tag2id = word_tag_map["tag2id"]

    print("推理模型：{}".format(model_name))
    model = load_model(model_path)
    if "BILSTM" in model_name:
        if model_name == "BILSTM_CRF":
            for_crf = True
            # model.model.bilstm.bilstm.flatten_parameters()  # remove warning
        else:
            for_crf = False
            # model.model.bilstm.flatten_parameters()  # remove warning


        word2id, tag2id = extend_maps(word2id, tag2id, for_crf=for_crf)
    pred_tag_list = model.predict(word_list, word_tag_map,for_crf)

    pred_tag_list = flatten_lists(pred_tag_list)
    print(pred_tag_list)
    return pred_tag_list


def predict_ensemble():
    dataset = args.dataset
    text = args.text

    data_dir = "./datasets/" + dataset + "/data/"
    model_dir = "./datasets/"+dataset + "/ckpts/"
    word_list = list(text.replace(" ", ""))


    model_list = [x for x in os.listdir(model_dir) if x.endswith(".pkl")]

    results = []
    for file_name in model_list:
        model_name = os.path.splitext(file_name.upper())[0]
        model_path = "./datasets/" + dataset + "/ckpts/" + model_name.lower() + ".pkl"

        #不同模型使用的有区别，所以加载新的模型时，重新加载
        word_tag_map = load_params(data_dir)
        word2id = word_tag_map["word2id"]
        tag2id = word_tag_map["tag2id"]

        if model_name == "BILSTM_CRF":
            for_crf = True
        else:
            for_crf = False

        if "BILSTM" in model_name:
            if model_name == "BILSTM_CRF":
                for_crf = True
            else:
                for_crf = False
            word2id, tag2id = extend_maps(word2id, tag2id, for_crf=for_crf)

        # print("推理模型：{}".format(model_name))
        model = load_model(model_path)
        res = model.predict(word_list, word_tag_map,for_crf)
        results.append(res)

    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tag_list = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tag_list.append(ensemble_tag)

    print(pred_tag_list)
    return pred_tag_list


if __name__ == "__main__":
    if args.model != "ensemble":
        predict()
    else:
        predict_ensemble()