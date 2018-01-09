from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import numpy as np
from sklearn.svm import LinearSVC


def report(results, n_top=3):
    """
    Utility function to report best scores
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#
    sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    :param results:
    :param n_top:
    :return:
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def binarize_targets(targets, positive_label):
    return [0 if x != positive_label else 1 for x in targets]


def main():
    # Categories to process http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
    categories = [
        'alt.atheism', 'talk.religion.misc',
        'comp.graphics', 'sci.space'
    ]
    # categories = None  # uncomment if you want to analyze 20 categories

    newsgroups_all = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    #  Text preprocessing
    vectorizer = TfidfVectorizer(stop_words='english') # remove english stop words ('in', 'at', ect.)
    vectors = vectorizer.fit_transform(newsgroups_all.data)
    binary_target = binarize_targets(newsgroups_all.target, positive_label=1)
    X_train, X_test, y_train, y_test = train_test_split(vectors, binary_target)

    rbm = BernoulliRBM(verbose=True, n_iter=3)
    svm = LinearSVC()
    classifier = Pipeline([('rmb', rbm), ('svm', svm)])
    classifier.fit(X_train, y_train)

    print("BernoulliRBM + LinearSVC: Test score {}".format(classifier.score(X_train, y_train)))
    print("BernoulliRBM + LinearSVC: Train score {}".format(classifier.score(X_test, y_test)))


if __name__ == "__main__":
    main()
