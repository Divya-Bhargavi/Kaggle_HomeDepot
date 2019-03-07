from sklearn.base import BaseEstimator, clone, RegressorMixin, TransformerMixin
from sklearn.model_selection import KFold
from Feature_Engineering.feature_engineering import *


class StackedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds
    
    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        for i, clf in enumerate(self.regressors):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)
                train_data = [X.iloc[idx,] for idx in train_idx]
                train_target = [y.iloc[idx,] for idx in train_idx]
                val_data = [X.iloc[idx,] for idx in holdout_idx]
                instance.fit(train_data, train_target)
                y_pred = instance.predict(val_data)
                out_of_fold_predictions[holdout_idx, i] = y_pred.reshape(out_of_fold_predictions[holdout_idx, i].shape)

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)

    
class CleanedTerms(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        cleaned_terms = [' '.join(tokenize(search_term))
                         for search_term in X['search_term']]
        X['cleaned_terms'] = cleaned_terms
        return X


class StemmedTerms(BaseEstimator, TransformerMixin):
    def __init__(self, new_col_name, orig_col_name):
        self.new_col_name = new_col_name
        self.orig_col_name = orig_col_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X[self.new_col_name] = [' '.join(stemmed(tokenize(word)))
                                for word in X[self.orig_col_name]]
        return X


class LemmatizedTerms(BaseEstimator, TransformerMixin):
    def __init__(self, new_col_name, orig_col_name):
        self.new_col_name = new_col_name
        self.orig_col_name = orig_col_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X[self.new_col_name] = [
            ' '.join(lemmatized(tokenize(word))) for word in X[self.orig_col_name]]
        return X


class Length(BaseEstimator, TransformerMixin):
    def __init__(self, new_col_name, data_col_name):
        self.new_col_name = new_col_name
        self.data_col_name = data_col_name

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        df = pd.DataFrame(get_length(list(X[self.data_col_name])))
        df.columns = [self.new_col_name]
        return df


class CountWords(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, func):
        self.col_name = col_name
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(X[self.col_name].apply(self.func))


class FindTermsInCorpus(BaseEstimator, TransformerMixin):
    def __init__(self, terms, corpus):
        self.terms = terms
        self.corpus = corpus

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        return clean_term_in_doc(X[self.terms],
                                 X[self.corpus])


class FindNeighbors(BaseEstimator, TransformerMixin):
    def __init__(self, terms, dictionary, glove_file):
        self.terms = terms
        self.dictionary = dictionary
        self.glove_file = glove_file

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        wordlist, matrix = split_dictionary(self.glove_file)
        cleaned_set = unique_words(X[self.terms])
        find_nearest_neighbors('Data/glove_neighbour_no_w.txt',
                               cleaned_set, matrix, wordlist, self.dictionary)
        k_dict = build_dictionary('Data/glove_neighbour_no_w.txt')
        X['terms_neighbour'] = get_all_terms_neighbors(
            k_dict, list(X[self.terms]))
        return pd.DataFrame(X)


class FindNeighborsInCorpus(BaseEstimator, TransformerMixin):
    def __init__(self, terms_neighbour, corpus):
        self.terms_neighbour = terms_neighbour
        self.corpus = corpus

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(clean_term_in_doc(X[self.terms_neighbour], X[self.corpus]))


class FindColorInSearchTerm(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        attrib_col = self.attributes[self.attributes['name'].apply(
            lambda x: 'color' in str(x).lower())]
        attrib_col = attrib_col.groupby(
            'product_uid').agg(lambda x: x.tolist())
        attrib_col = attrib_col.drop('name', axis=1)
        attrib_col = attrib_col.reset_index()
        attrib_col = attrib_col.rename(columns={'value': 'color'})

        attrib_col['color'] = attrib_col['color'].apply(lambda x: ','.join(x))
        attrib_col['color'] = attrib_col['color'].apply(
            lambda x: ','.join(x.replace('/', '').replace(' ', ',').split(',')).replace(',,', ','))

        X = X.set_index('product_uid').join(
            attrib_col.set_index('product_uid'))
        X = X.reset_index()
        attrib_col = attrib_col.reset_index()
        X['color'].fillna('', inplace=True)
        X['search_term'].fillna('', inplace=True)
        X['color'] = X['color'].apply(lambda x: set(x.split(',')))

        color_in_search_term = []
        for i in range(len(X)):
            p = len(X['color'][i].intersection(
                X['search_term_split'][i]))
            color_in_search_term.append(p)

        return pd.DataFrame(color_in_search_term)


class MinLevensteinDistBrand(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        attr_brand = self.attributes[(self.attributes['name'].str.lower().str.contains(
            'brand') == True) & self.attributes['value'].notnull()]
        attr_brand = attr_brand.drop('name', axis=1)
        attr_brand = attr_brand.rename(columns={'value': 'brand'})
        attr_brand['product_uid'] = attr_brand['product_uid'].apply(
            lambda x: int(x))

        d = defaultdict(list)
        p = list(attr_brand['product_uid'])
        b = list(attr_brand['brand'])
        for i in range(len(p)):
            if p[i] not in d:
                d[p[i]] = tokenize(b[i])
            else:
                continue
        X['brand'] = X['product_uid'].apply(lambda x: d[x])
        X['brand'].fillna('', inplace=True)
        X['search_term'].fillna('', inplace=True)
        X['search_term_split'] = X['search_term'].apply(
            lambda x: x.split(' '))

        search_term_split = X['search_term_split'].tolist()
        brand = X['brand'].tolist()

        p = []
        for i in range(len(X)):
            q = []
            if len(search_term_split[i][0]) > 0:
                for j in range(len(search_term_split[i])):
                    for k in range(len(brand[i])):
                        if search_term_split[i][j] in brand[i][k]:
                            q.append(
                                (brand[i][k], brand[i][k]))
                            continue
                        elif search_term_split[i][j][0] == brand[i][k][0]:
                            q.append((search_term_split
                                      [i][j], brand[i][k]))
            p.append(q)

        l = []
        for i in range(len(p)):
            q = []
            for j in range(len(p[i])):
                q.append(distance(p[i][j][0], p[i][j][1]))
            l.append(q)

        m = []
        for q in l:
            if q == []:
                m.append(1000)
            else:
                m.append(min(q))

        return pd.DataFrame(m)


class MinLevensteinDistTitle(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        X['product_title_clean'] = X['product_title'].apply(
            lambda x: list(set(tokenize(x))))
        X['search_term'].fillna('', inplace=True)
        X['search_term_split'] = X['search_term'].apply(
            lambda x: x.split(' '))

        search_term_split = X['search_term_split'].tolist()
        product_title_clean = X['product_title_clean'].tolist()

        p = []
        for i in range(len(X)):
            q = []

            if len(search_term_split[i]) > 0:
                for j in range(len(search_term_split[i])):
                    for k in range(len(product_title_clean[i])):
                        if search_term_split[i][j] in product_title_clean[i][k]:
                            q.append(
                                (product_title_clean[i][k], product_title_clean[i][k]))
                            continue
                        elif search_term_split[i][j][0] == product_title_clean[i][k][0]:
                            q.append((search_term_split[i][j],
                                      product_title_clean[i][k]))
            p.append(q)

        l = []
        for i in range(len(p)):
            q = []
            for j in range(len(p[i])):
                q.append(distance(p[i][j][0], p[i][j][1]))
            l.append(q)

        m = []
        for q in l:
            if q == []:
                m.append(1000)
            else:
                m.append(min(q))

        return pd.DataFrame(m)

    
class TFIDFSearchIntersection(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, **transform_params):
        X = find_n_tfidf_highest_scores(X, 5)
        X['search_term_split'] = X['search_term'].apply(
            lambda x: tokenizer(x))
        p = X['search_term_split'].tolist()
        q = X['tfidf'].tolist()
        l = []
        for i in range(len(p)):
            l.append(len(set(p[i]).intersection(set(q[i]))))
        return pd.DataFrame(l)
    

class JaccardIndex(BaseEstimator, TransformerMixin):
    def __init__(self, corpus, terms):
        self.corpus = corpus
        self.terms = terms

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        return calculate_jaccard_index(X[self.corpus], X[self.terms])


class Entropy(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(calculate_entropy(letter_prob(X[self.col_name])))


class LCS(BaseEstimator, TransformerMixin):
    def __init__(self, corpus, terms):
        self.corpus = corpus
        self.terms = terms

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        pool = Pool()
        lcs = pool.starmap(longest_common_subsequence_parallelized, zip(
            list(X[self.terms]), list(X[self.corpus])))
        pool.close()
        return pd.DataFrame(lcs)


class Jaro(BaseEstimator, TransformerMixin):
    def __init__(self, corpus, terms):
        self.corpus = corpus
        self.terms = terms

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        pool = Pool()
        jaro_score = pool.starmap(
            getJaroScoreOnDocs, zip(X[self.terms], X[self.corpus]))
        pool.close()
        return pd.DataFrame(jaro_score)


class SW_Score(BaseEstimator, TransformerMixin):
    def __init__(self, corpus, terms):
        self.corpus = corpus
        self.terms = terms

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        pool = Pool()
        sw_scores = pool.starmap(create_SW_score_col_parallelized, zip(
            list(X[self.terms]), list(X[self.corpus])))
        pool.close()
        return pd.DataFrame(sw_scores)


class NCD(BaseEstimator, TransformerMixin):
    def __init__(self, terms, corpus):
        self.terms = terms
        self.corpus = corpus

    def fit(self, X, y=None):
        return self

    def transform(self, X, **transform_params):
        pool = Pool()
        ncd_query_title = pool.starmap(compute_ncd_parallelized, zip(
            list(X[self.terms]), list(X[self.corpus])))
        pool.close()
        return pd.DataFrame(ncd_query_title)


class CountAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X = add_num_attrib_per_prod_column(X, self.attributes)
        return pd.DataFrame(X['num_attrib'])