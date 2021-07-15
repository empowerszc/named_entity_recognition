from sklearn_crfsuite import CRF

from utils.model_util import sent2features


class CRFModel(object):
    def __init__(self,
                 vocab_size, out_size, for_crf,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False
                 ):

        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)
    # train_data, dev_data, word_tag_map
    def train(self, train_data, dev_data, word_tag_map, *args):
        sentences = train_data[0]
        tag_lists = train_data[1]

        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    # test_data, word_tag_map
    def test(self, test_data, word_tag_map, *args):
        sentences = test_data[0]
        tag_lists = test_data[1]

        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists, tag_lists


    def predict(self, sentence, word_tag_map, *args):
        sentences = [sentence]
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists
