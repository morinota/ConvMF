"""sklearn.feature_extraction.text.CountVectorizerを用いて、
テキスト情報をベクトルに変換します。
csr_matrixクラスについてdocument:https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html?highlight=csr_matrix#scipy.sparse.csr_matrix
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def get_count_vectors(corpus: List[List[str]], **params_count_vectorizer) -> Tuple[csr_matrix, CountVectorizer]:
    """text(tokenのList)のListを受け取り、
    sklearn.feature_extraction.text.CountVectorizerを用いて、
    テキスト情報をベクトルに変換します。
    CountVectorizerのオプション指定は、元論文を参考にする.
    """
    vectorizer = CountVectorizer(**params_count_vectorizer)
    vectorizer.fit(raw_documents=corpus)
    count_vectors = vectorizer.transform(raw_documents=corpus)
    token_list = vectorizer.get_feature_names_out()
    token_idx_mapping = {idx: token for idx, token in enumerate(token_list)}
    return count_vectors, vectorizer


def get_tfidf_vectors(
    count_vectors: csr_matrix,
    **params_tfidf_transformer,
) -> Tuple[csr_matrix, TfidfTransformer]:
    """count vectorsを受け取り、tf-idf vectorsに変換して返す."""
    vectorizer = TfidfTransformer(**params_tfidf_transformer)
    vectorizer.fit(count_vectors)
    tfidf_vectors = vectorizer.transform(count_vectors)
    return tfidf_vectors, vectorizer
