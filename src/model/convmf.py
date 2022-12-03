
from typing import List

from torch import embedding


import numpy as np
from model.matrix_factorization import RatingData, MatrixFactrization
from model.model_cnn_nlp import CNN_NLP


class ConvMF(object):
    def __init__(self,
                 ratings: List[RatingData],
                 filter_windows: List[int],
                 max_sentence_length: int,
                 item_descriptions: List[np.ndarray],
                 n_word,
                 n_out_channel=100,
                 dropout_ratio=0.5,
                 n_factor=300,
                 user_lambda=0.001,
                 item_lambda=0.001,
                 mf: MatrixFactrization = None) -> None:
        
        self.n_factor = n_factor
        self.item_descriptions = item_descriptions
        n_item = len(item_descriptions)
        self.mf = mf
        if self.mf is None:
            self.mf = MatrixFactrization(ratings, n_factor,
                                         user_lambda, item_lambda, n_item)
                                
        
        # model architecture
        self.convolution = CNN_NLP()

    def __call__(self,x,y=None,train=True) -> Any:
        if train:
            loss = self.convolution(x, y)
            return loss
        else:
            return self.convolution(x)

    def predict(self, users:List[int], items:List[int])->List[np.ndarray]:
        # item factor Vを取得
        item_factors = self.convolution(
            x=np.array([self.item_descriptions[i] for i in items])
            )
        # user factor Uを所得
        user_factors:List[np.ndarray] = []
        # r_hatを計算
        predictions = []

        return predictions

    def get_item_factors(self,items:List[int])->List[np.ndarray]:
        # itemに対応するitem factor vectorのlistを取得
        item_factors = self.convolution(
            x=np.array([self.item_descriptions[i] for i in items])
            )
        return item_factors

    def fit_mf(self, n_trial =3):
        self.mf.fit(n_trial=n_trial)

    def update_latent_factor(self, n_trial=3):

        self.mf.fit(n_trial=n_trial, additional=self._embedding())
    
    def _embedding(self)->List[np.ndarray]:
        # document latent factorのListを作成する関数
        embedding = [self.convolution(d) for d in self.item_descriptions]
        return embedding