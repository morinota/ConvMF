import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary

from src.config import MyConfig
from src.dataclasses.item_description import ItemDescription
from src.dataclasses.rating_data import RatingLog
from src.model.matrix_factorization import MatrixFactrization


class ItemDescrptionPreparer:
    def __init__(self, descriptions_csv_path: str) -> None:
        self.descriptions_csv_path = descriptions_csv_path

    def load(self, max_token_num_in_text: Optional[int] = None) -> List[ItemDescription]:
        """CovMFに入力するDocument情報(X_j)を作成する関数。

        Args:
            max_sentence_length (int, optional):
            Document情報の最大長さ(Tokenの数)

        Returns:
            Tuple[np.ndarray, np.ndarray, int]:
            - アイテムid のndarray
            - アイテムidに対応するdocument contextのndarray(wordをindex化してる)
            -
        """
        descriptions_df = pd.read_csv(self.descriptions_csv_path).rename(columns={"movie": "item_id"})
        text_series = descriptions_df["description"]

        tokens_in_texts = self._tokenize_text_series(text_series)

        self.dictionary = self._register_tokens_to_dict(tokens_in_texts)

        # TODO: unknown_word_indexをeos_id(idxの最後尾+1)にする必要ある?
        self.n_all_tokens_registerd = len(self.dictionary.keys())

        token_idx_array_in_texts = self._get_index_each_token(
            tokens_in_texts,
            self.dictionary,
            idx_for_unknown_token=self.n_all_tokens,  # tokenのidxの最後尾+1をunkown_token用のidxに
        )

        self.max_token_num_in_text = max([len(token_idx_array) for token_idx_array in token_idx_array_in_texts])

        token_idx_array_in_texts_padded = self._padding_all_discriptions(
            token_idx_array_in_texts,
            self.max_token_num_in_text,
            idx_for_unknown_token=self.n_all_tokens,  # tokenのidxの最後尾+1をunkown_token用のidxに
        )

        # change types
        token_idx_array_in_texts_padded = [
            token_idx_array.astype(np.int32) for token_idx_array in token_idx_array_in_texts_padded
        ]
        item_id_series = descriptions_df["id"].astype(np.int32)

        return [
            ItemDescription(
                item_id,
                original_text,
                tokens,
                token_indices,
            )
            for item_id, original_text, tokens, token_indices in zip(
                item_id_series.to_list(),
                text_series.to_list(),
                tokens_in_texts,
                token_idx_array_in_texts_padded,
            )
        ]

    def _tokenize_text_series(self, text_series: pd.Series) -> List[List[str]]:
        """各アイテムに対応するテキストをtokenizeする
        英文なので半角スペースでtokenizeする
        str.strip()：stringの両端の指定した文字を削除する.
        defaultは空白文字(改行\nや全角スペース\u3000やタブ\tなどが空白文字とみなされ削除)
        """
        text_series_removed_edge_space = text_series.apply(lambda x: x.strip())
        text_series_tokenized = text_series_removed_edge_space.apply(lambda x: x.split())
        return text_series_tokenized.to_list()

    def _register_tokens_to_dict(self, tokens_in_texts: List[List[str]]) -> Dictionary:
        """text_series_tokenizedの単語(token)をDictionaryオブジェクト(# id: tokenのdict)に登録し、コーパスを作る
        (ここでいうコーパスとは、文書内のtokenを集めたデータ。)

        Parameters
        ----------
        tokens_in_texts : List[List[str]]
            ex. ->[['human', 'interface', 'computer'],['survey', 'user', 'computer', 'system'],...]

        Returns
        -------
        Tuple[Dictionary, int]
            _description_
        """
        dictionary = Dictionary(tokens_in_texts)
        dictionary.filter_extremes(
            no_below=5,  # n個以上のtextに登場したtokenだけ残す
            no_above=0.5,  # 全textの0.5=50%よりも多く登場してるtokenを消す
        )  # tokenの出現回数に応じてfiltering
        doc_frequencies_each_token = dictionary.dfs
        count_frequencies_each_token = dictionary.cfs

        return dictionary

    def _get_index_each_token(
        self,
        tokens_in_texts: List[List[str]],
        dict_obj: Dictionary,
        idx_for_unknown_token: int,
    ) -> List[np.ndarray]:
        """各text内の各tokenを、dictionaryに登録された通し番号(index)で置き換える(idx_for_unknown_tokenはdictに存在しないtoken)
        ex) ['this','is','sparta','I','am','sparta'] -> [6, 4, 5, idx_for_unknown_token, idx_for_unknown_token, 5]
        Parameters
        ----------
        tokens_in_texts : List[List[str]]
            _description_
        dict_obj : Dictionary
            _description_
        eos_id : int
            _description_

        Returns
        -------
        List[np.ndarray]
            各text内の各tokenを、dictionaryに登録された通し番号(index)に変換した配列(intのndarray)
        """
        token_indices_in_texts = [
            dict_obj.doc2idx(tokens_in_a_text, unknown_word_index=idx_for_unknown_token)
            for tokens_in_a_text in tokens_in_texts
        ]

        # token_indices_in_textsを取り除きつつ、List[List[int]]をList[ndarray]に変換
        token_idx_array_in_texts = [
            np.array([token_idx for token_idx in token_indices if token_idx != idx_for_unknown_token])
            for token_indices in token_indices_in_texts
        ]
        return token_idx_array_in_texts

    def _padding_all_discriptions(
        self,
        token_idx_array_in_texts: List[np.ndarray],
        max_text_length: int,
        idx_for_unknown_token: int,
    ) -> List[np.ndarray]:
        """padding: 全てのitem descriptionのindicesの長さをmax_text_lengthに揃える
        paddingした要素のidxはidx_for_unknown_wordで穴埋めする.
        """
        self.padding_word_idx = idx_for_unknown_token
        token_idx_array_in_texts_slised = [
            token_idx_array[:max_text_length] for token_idx_array in token_idx_array_in_texts
        ]
        token_idx_array_in_texts_padded = [
            np.pad(
                token_idx_array,
                pad_width=(0, max_text_length - len(token_idx_array)),
                mode="constant",
                constant_values=(0, idx_for_unknown_token),
            )
            for token_idx_array in token_idx_array_in_texts_slised
        ]
        return token_idx_array_in_texts_padded

    @property
    def n_all_tokens(self):
        return self.n_all_tokens_registerd

    @property
    def max_length_in_one_description(self) -> int:
        return self.max_token_num_in_text

    @property
    def word2idx_mapping(self) -> Dict[str, int]:
        word2idx_mapping = {word: idx for idx, word in self.dictionary.items()}
        word2idx_mapping["<pad>"] = self.padding_word_idx # dictionaryにはpaddingが未登録の為追加
        return word2idx_mapping

    @property
    def idx2word_mapping(self) -> Dict[int, str]:
        return dict(self.dictionary)


if __name__ == "__main__":
    item_description_preparer = ItemDescrptionPreparer(MyConfig.descriptions_path)
    item_descriptions = item_description_preparer.load()
    n_all_tokens = item_description_preparer.n_all_tokens
    print(item_descriptions[1])
    print(n_all_tokens)
