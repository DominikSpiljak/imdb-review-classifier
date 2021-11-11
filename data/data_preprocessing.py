from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfPreprocessor:
    def __init__(self, *args, **kwargs):
        self.vectorizer = TfidfVectorizer(*args, **kwargs)

    def __call__(self, train_set, test_set):
        print("Running tfidf on corpus...")
        train_set_vectorized = self.vectorizer.fit_transform(train_set)
        test_set_vectorized = self.vectorizer.transform(test_set)

        return train_set_vectorized, test_set_vectorized
