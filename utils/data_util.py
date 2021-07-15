from os.path import join
from codecs import open
import pickle

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def save_params(data_dir, data):
    with open(join(data_dir,"data.pkl"), "wb") as fopen:
        pickle.dump(data, fopen)

def load_params(data_dir):
    with open(join(data_dir,"data.pkl"), "rb") as fopen:
        data_map = pickle.load(fopen)
    return data_map

def build_corpus(action, data_dir=data_dir):
    """读取数据"""
    assert action  in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, action+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

        return word_lists, tag_lists