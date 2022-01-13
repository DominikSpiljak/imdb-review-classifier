from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
import torch
import numpy as np


class TfIdfPreprocessor:
    def __init__(self, *args, **kwargs):
        self.vectorizer = TfidfVectorizer(*args, **kwargs)

    def __call__(self, train_set, val_set, test_set):
        print("Running tfidf on corpus...")
        train_set_vectorized = self.vectorizer.fit_transform(train_set)
        val_set_vectorized = self.vectorizer.transform(val_set)
        test_set_vectorized = self.vectorizer.transform(test_set)

        return (
            self.csr_matrix_to_sparse_tensor(train_set_vectorized),
            self.csr_matrix_to_sparse_tensor(val_set_vectorized),
            self.csr_matrix_to_sparse_tensor(test_set_vectorized),
        )

    def csr_matrix_to_sparse_tensor(self, csr_matrix):
        matrix_coo = csr_matrix.tocoo()

        return torch.sparse.FloatTensor(
            torch.LongTensor([matrix_coo.row.tolist(), matrix_coo.col.tolist()]),
            torch.FloatTensor(matrix_coo.data.astype(np.float)),
        )


class RNNPreprocessor:
    def __init__(self, max_seq_len, *args, **kwargs):
        self.vectorizer = CountVectorizer(*args, **kwargs)
        self.max_seq_len = max_seq_len

    def __get_mappings(self, train_set):
        self.vectorizer.fit(train_set)
        words = self.vectorizer.get_feature_names()
        self.word_to_ind = {}
        index = 2  # 0 is a PAD and 1 is UNK
        for word in words:
            self.word_to_ind[word] = index
            index += 1
        self.ind_to_word = {v: k for k, v in self.word_to_ind.items()}

    def __lower_split_and_vectorize(self, entry):
        entry = entry.lower()
        entry = [
            self.word_to_ind.get(word.strip(string.punctuation), 1)
            for word in entry.split()
        ][: self.max_seq_len]
        return entry

    def __call__(self, train_set, val_set, test_set):
        print("Running vectorization on corpus...")
        self.__get_mappings(train_set)

        return (
            list(map(self.__lower_split_and_vectorize, train_set)),
            list(map(self.__lower_split_and_vectorize, val_set)),
            list(map(self.__lower_split_and_vectorize, test_set)),
        )
