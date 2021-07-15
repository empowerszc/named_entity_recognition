import sys
import time


import argparse
from utils.data_util import build_corpus, build_map ,save_params,load_params
from utils.preprocess_util import save_model,load_model,flatten_lists,extend_maps
from utils.evaluating import Metrics
from utils.config import MODEL_LIST
from models.model_factory import getModel

parser = argparse.ArgumentParser(description="Named Entity Recognition")
parser.add_argument("action", type=str, help="train | test | predict")
parser.add_argument("--model", type=str, default="BILSTM_CRF")
parser.add_argument("--dataset", type=str, default="ResumeNER")
parser.add_argument("--text", type=str, )
args = parser.parse_args()


def main():
    # 没有多的其他参数，读取配置文件
    # if len(sys.argv) ==1:
    model_name = args.model

    assert  model_name in MODEL_LIST, "模型不存在"

    if model_name == "BILSTM_CRF":
        # model_name == "BILSTM"
        for_crf = True
    else:
        for_crf = False

    eval = True

    # 准备数据集
    dataset = args.dataset
    data_dir = "./datasets/"+dataset +"/data/"
    train_word_lists, train_tag_lists = build_corpus("train", data_dir=data_dir)
    dev_word_lists, dev_tag_lists = build_corpus("dev", data_dir=data_dir)
    test_word_lists, test_tag_lists = build_corpus("test", data_dir=data_dir)

    data = [(train_word_lists,train_tag_lists), (dev_word_lists, dev_tag_lists), (test_word_lists, test_tag_lists)]

    # 字典数据持久化
    word2id = build_map(train_word_lists)
    tag2id = build_map(train_tag_lists)
    word_tag_map = {
        "word2id":word2id,
        "tag2id":tag2id,
    }
    # 将数据存储起来
    save_params(data_dir, word_tag_map)

    print("正在训练评估{}模型...".format(model_name))
    train_eval(model_name, dataset, data, word_tag_map, for_crf, eval=eval)


def train_eval(model_name,dataset, data, word_tag_map, for_crf, eval=True):
    # 一般都是3个
    if len(data) == 3:
        train_data = data[0]
        dev_data = data[1]
        test_data = data[2]
    elif len(data) ==2:
        train_data = data[0]
        dev_data = None
        test_data = data[1]


    model = getModel(model_name, word_tag_map, for_crf)
    saved_path = "./datasets/"+dataset + "/ckpts/" + model_name.lower()+".pkl"

    start = time.time()
    model.train(train_data, dev_data, word_tag_map, for_crf)

    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))

    save_model(model, saved_path)

    if eval == True:
        print("评估模型{}".format(model_name))

        pred, target_tag_list = model.test(test_data, word_tag_map, for_crf)

        metrics = Metrics(target_tag_list, pred, remove_O=False) # 在评估的时候是否去除O标记
        metrics.report_scores()
        metrics.report_confusion_matrix()


def test():
    model_name = args.model
    dataset = args.dataset
    data_dir = "./datasets/"+dataset +"/data/"
    model_path = "./datasets/"+dataset + "/ckpts/" + model_name.lower()+".pkl"

    word_tag_map = load_params(data_dir)
    word2id = word_tag_map["word2id"]
    tag2id = word_tag_map["tag2id"]

    if "BILSTM" in model_name:
        if model_name == "BILSTM_CRF":
            for_crf = True
        else:
            for_crf = False
        word2id, tag2id = extend_maps(word2id, tag2id, for_crf=for_crf)

    test_word_lists, test_tag_lists = build_corpus("test", data_dir=data_dir)
    test_data = (test_word_lists, test_tag_lists)



    print("评估{}模型".format(model_name))
    print(model_path)
    model = load_model(model_path)

    pred, target_tag_list = model.test(test_data, word_tag_map,for_crf)

    metrics = Metrics(target_tag_list, pred, remove_O=False)  # 在评估的时候是否去除O标记
    metrics.report_scores()
    metrics.report_confusion_matrix()


if __name__ == "__main__":
    if args.action == "train":
        main()
    elif args.action == "test":
        test()
