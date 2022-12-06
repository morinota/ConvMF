import time
from src.text_cnn_test.cnn_nlp_model.train_nlp_cnn import train
from src.text_cnn_test.utils.dataloader import create_data_loaders

import torch
from sklearn.model_selection import train_test_split
from src.config import MyConfig
from src.dataclasses.item_description import ItemDescription
from src.model.matrix_factorization import MatrixFactrization
from src.model.model_cnn_nlp import CnnNlpModel, initilize_cnn_nlp_model
from src.utils.item_description_preparer import ItemDescrptionPreparer
from src.utils.rating_log_loader import RatingLogReader
from src.utils.word_vector_preparer import WordEmbeddingVector


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
    rating_log_reader = RatingLogReader(ratings_csv_path=MyConfig.ratings_path)
    rating_logs = rating_log_reader.load()
    print(len(rating_logs))

    max_sentence_length = 300  # 300 token(word)
    item_description_prepaper = ItemDescrptionPreparer(MyConfig.descriptions_path)
    item_descriptions = item_description_prepaper.load(max_sentence_length)
    token2idxs_mapping = item_description_prepaper.word2idx_mapping
    token_indices_array = ItemDescription.merge_token_indices_of_descriptions(
        item_descriptions,
    )

    embedding_vectors = WordEmbeddingVector.load_pretrained_vectors(
        token2idxs_mapping,
        MyConfig.fast_text_path,
        padding_word="<pad>",
    )

    # check the device (GPU|CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    mf_obj = MatrixFactrization(
        rating_logs=rating_logs,
        n_factor=10,
        n_item=len(item_descriptions),  # 登録されているアイテム数を追加しておく(rating_logsに含まれてない可能性がある)
    )

    # train test split
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        token_indices_array, mf_obj.item_latent_vectors, test_size=0.1, random_state=42
    )

    # Load data to Pytorch DataLoader
    train_dataloader, val_dataloader = create_data_loaders(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        train_labels=train_labels,
        val_labels=val_labels,
        batch_size=50,
    )

    # 一つの値を取り出す
    data = next(iter(train_dataloader))
    # 結果の確認
    print(data)  # tensor([1, 2])

    cnn_nlp, optimizer = initilize_cnn_nlp_model(
        pretrained_embedding=embedding_vectors.to_tensor(),
        freeze_embedding=True,  # fastText で事前学習された単語ベクトルが使われ、学習中は凍結される。
        learning_rate=0.25,
        dropout=0.5,
        device=device,
        output_dimension=mf_obj.n_factor,
    )

    cnn_nlp_trained = train(
        model=cnn_nlp,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        epochs=20,
        device=device,
    )

    # mf_obj.fit(n_trial=5, document_vectors=None)


if __name__ == "__main__":
    main()
