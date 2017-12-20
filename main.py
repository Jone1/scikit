from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import numpy as np

# Utility function to report best scores
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


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
    X_train, X_test, y_train, y_test = train_test_split(vectors, newsgroups_all.target)

    # Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Logistic regression: Test score {}".format(model.score(X_train, y_train)))
    print("Logistic regression: Train score {}".format(model.score(X_test, y_test)))

    # Logistic Regression + BernoulliRBM
    rbm = BernoulliRBM(
        random_state=0, verbose=True,
        n_iter=3,
        learning_rate=0.06,
        n_components=10,
    )
    logistic = LogisticRegression()

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    classifier.fit(vectors, newsgroups_all.target)

    print("BernoulliRBM + Logistic regression: Test score {}".format(classifier.score(X_train, y_train)))
    print("BernoulliRBM + Logistic regression: Train score {}".format(classifier.score(X_test, y_test)))

    # Previous example gives poor results, so: RandomSearch (Logistic Regression + BernoulliRBM)
    param_dist = {
        'rbm__learning_rate': [0.001, 0.05, 0.2, 0.8],
        'rbm__n_components': [10, 100, 1000],
        'rbm__n_iter': [5, 10, 100],
    }
    random_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_dist, n_iter=20, verbose=True)
    random_search.fit(X_train, y_train)
    report(random_search.cv_results_)

if __name__ == "__main__":
    main()
