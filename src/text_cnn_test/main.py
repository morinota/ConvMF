import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from cnn_nlp_model.train_nlp_cnn import set_seed, train
from sklearn.model_selection import train_test_split

from src.config import MyConfig
from src.dataclasses.item_description import ItemDescription
from src.model.cnn_nlp_model import initilize_cnn_nlp_model
from src.utils.item_description_preparer import ItemDescrptionPreparer
from src.utils.word_vector_preparer import WordEmbeddingVector
from utils.dataloader import create_data_loaders
from utils.pretrained_vec import load_pretrained_vectors

# nltk.download('all')
from utils.tokenizes import conduct_tokenize, encode

# FAST_TEXT_PATH = r'..\data\fastText\crawl-300d-2M.vec\crawl-300d-2M.vec'


def load_word_vector():
    URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
    FILE = "fastText"

    if os.path.isdir(FILE):
        print("fastText exists.")
    else:
        print("please download fastText.")


def main():
    load_word_vector()

    max_sentence_length = 300  # 300 token(word)
    item_description_prepaper = ItemDescrptionPreparer(MyConfig.descriptions_path)
    item_descriptions = item_description_prepaper.load(max_sentence_length)
    word2idx_mapping = item_description_prepaper.word2idx_mapping
    token_indices_array = ItemDescription.merge_token_indices_of_descriptions(
        item_descriptions,
    )

    # 今回は実装テストなので、labelを適当に作成
    labels = np.array(
        [0] * len(item_descriptions[: len(item_descriptions) % 2])
        + [1] * len(item_descriptions[len(item_descriptions) % 2 :])
    )

    print(f"the num of texts data is {len(item_descriptions)}, and the num of labels is {len(labels)}.")

    print(f"the num of vocabrary is {len(word2idx_mapping) - 2}")
    print(f"the shape of input_ids is {token_indices_array.shape}")

    # Load pretrained embedding vectors
    embedding_vectors = WordEmbeddingVector.load_pretrained_vectors(
        word2idx_mapping,
        MyConfig.fast_text_path,
        padding_word="<pad>",
    )
    print(f"[LOG]the shape of embedding_vectors is {embedding_vectors.vectors.shape}")

    # train test split
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        token_indices_array, labels, test_size=0.1, random_state=42
    )

    # Load data to Pytorch DataLoader
    train_dataloader, val_dataloader = create_data_loaders(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        train_labels=train_labels,
        val_labels=val_labels,
        batch_size=50,
    )

    # check the device (GPU|CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Log]device : {device}")

    # CNN-static: fastText pretrained word vectors are used and freezed during training.
    # fastText で事前学習された単語ベクトルが使われ、学習中は凍結される。
    set_seed(42)
    cnn_nlp, optimizer = initilize_cnn_nlp_model(
        pretrained_embedding=embedding_vectors.to_tensor(),
        freeze_embedding=True,
        learning_rate=0.25,
        dropout=0.5,
        device=device,
        output_dimension=2,
    )

    cnn_nlp = train(
        model=cnn_nlp,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=20,
        device=device,
    )

    document_latent_vectors = cnn_nlp.predict()

    # # CNN-rand: Word vectors are randomly initialized(Word vectorの初期値をランダムにしたVer.)
    # set_seed(42)
    # cnn_rand, optimizer = initilize_model(
    #     vocab_size=len(word2idx),
    #     embed_dim=300,
    #     learning_rate=0.25,
    #     dropout=0.5, device=device
    # )

    # cnn_rand = train(model=cnn_rand,
    #                  optimizer=optimizer,
    #                  train_dataloader=train_dataloader,
    #                  val_dataloader=val_dataloader,
    #                  epochs=20, device=device
    #                  )

    # # CNN-non-static: fastText pretrained word vectors are fine-tuned during training.
    # # fastText 事前学習された単語ベクトルが使われ、学習中に微調整される
    # set_seed(42)
    # cnn_non_static, optimizer = initilize_model(
    #     pretrained_embedding=embeddings,
    #     freeze_embedding=False,
    #     learning_rate=0.25,
    #     dropout=0.5, device=device
    # )
    # cnn_non_static = train(model=cnn_non_static,
    #                        optimizer=optimizer,
    #                        train_dataloader=train_dataloader,
    #                        val_dataloader=val_dataloader,
    #                        epochs=20,
    #                        device=device
    #                        )


if __name__ == "__main__":
    main()
