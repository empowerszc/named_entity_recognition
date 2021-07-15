from models.model.hmm import HMM as HMM_Model
from models.model.crf import CRFModel as CRF_Model
from models.model.bilstm_crf import BILSTM_Model
from models.model.bilstm_crf import BILSTM_Model as BILSTM_CRF_Model

from utils.preprocess_util import extend_maps



def getModel(model_name, word_tag_map, for_crf):

    word2id = word_tag_map["word2id"]
    tag2id = word_tag_map["tag2id"]
    if "BILSTM" in model_name:
        word2id, lstm_tag2id = extend_maps(word2id, tag2id, for_crf=for_crf)
    vocab_size = len(word2id)
    out_size = len(tag2id)
    model = eval(model_name + "_Model(vocab_size, out_size, for_crf)")
    return model