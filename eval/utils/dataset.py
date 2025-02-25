import json



class Dataset:
    def __init__(self, path_src, path_tgt):
        file = open(path_src, 'r', encoding='utf-8')
        self.data_src = json.load(file)

        file = open(path_tgt, 'r', encoding='utf-8')
        self.data_tgt = json.load(file)

    def get_data(self):
        return (self.data_src, self.data_tgt)