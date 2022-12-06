import time

from src.config import MyConfig
from src.model.matrix_factorization import MatrixFactrization
from src.model.model_cnn_nlp import CnnNlpModel, initilize_cnn_nlp_model
from src.utils.item_description_preparer import ItemDescrptionPreparer
from src.utils.rating_log_loader import RatingLogReader


def train_convmf(batch_size: int, n_epoch: int, n_sub_epoch: int, n_out_channel: int):
    """_summary_

    Parameters
    ----------
    batch_size : int
        _description_
    n_epoch : int
        _description_
    n_sub_epoch : int
        _description_
    n_out_channel : int
        _description_
    """

    rating_log_reader = RatingLogReader(ratings_csv_path=MyConfig.ratings_path)
    rating_logs = rating_log_reader.load()

    filter_windows = [3, 4, 5]  # 窓関数の設定
    max_sentence_length = 300  # 300 token(word)

    item_description_prepaper = ItemDescrptionPreparer(MyConfig.descriptions_path)
    item_descriptions = item_description_prepaper.load(max_sentence_length)
    n_token = item_description_prepaper.n_all_tokens

    n_factor = 300
    dropout_ratio = 0.5
    user_lambda = 10
    item_lambda = 100


def main():
    # rating_log_reader = RatingLogReader(ratings_csv_path=MyConfig.ratings_path)
    # rating_logs = rating_log_reader.load()
    # print(len(rating_logs))

    max_sentence_length = 300  # 300 token(word)
    item_description_prepaper = ItemDescrptionPreparer(MyConfig.descriptions_path)
    item_descriptions = item_description_prepaper.load(max_sentence_length)
    token2idxs_mapping = item_description_prepaper.word2idx_mapping
    idx2token_mapping = item_description_prepaper.idx2word_mapping
    max_token_num_in_description = item_description_prepaper.max_length_in_one_description
    print(token2idxs_mapping)
    # print(idx2token_mapping)
    print(max_token_num_in_description)

    # mf_obj = MatrixFactrization(rating_logs=rating_logs, n_factor=10)

    # cnn_nlp, optimizer = initilize_cnn_nlp_model()

    # mf_obj.fit(n_trial=5, document_vectors=None)


if __name__ == "__main__":
    main()
