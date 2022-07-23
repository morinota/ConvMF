from typing import List
import pandas as pd
import os
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
# nltk.download('all')
from tokenizes import conduct_tokenize, encode
from pretrained_vec import load_pretrained_vectors
from model_cnn_nlp import initilize_model
from train_nlp_cnn import train, set_seed
from dataloader import create_data_loaders


TEXT_FILE = r'..\data\descriptions.csv'
FAST_TEXT_PATH = r'..\data\fastText\crawl-300d-2M.vec\crawl-300d-2M.vec'


def load_data():
    texts_df = pd.read_csv(TEXT_FILE)
    return texts_df


def load_word_vector():
    URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
    FILE = "fastText"

    if os.path.isdir(FILE):
        print("fastText exists.")
    else:
        print('please download fastText.')


def main():
    load_word_vector()

    texts_df = load_data()
    print(texts_df.head())

    # 文章をList[List[str]]として取得
    texts = texts_df['description'].to_list()

    # 今回は実装テストなので、labelを適当に作成
    labels = np.array(
        [0]*len(texts[:len(texts) % 2])
        + [1]*len(texts[len(texts) % 2:])
    )

    # データのサイズの確認
    print(
        f'the num of texts data is {len(texts)}, and the num of labels is {len(labels)}.')

    # Tokenize, build vocabulary, encode tokens
    print('Tokenizing...\n')
    tokenized_texts, word2idx, max_len = conduct_tokenize(texts=texts)
    print(f'the num of vocabrary is {len(word2idx) - 2}')
    print(f'max len of texts is {max_len}')
    input_ids = encode(tokenized_texts, word2idx, max_len)
    print(f'the shape of input_ids is {input_ids.shape}')

    # Load pretrained vectors
    embeddings = load_pretrained_vectors(word2idx, FAST_TEXT_PATH)
    print(f'the shape of embedding_vectors is {embeddings.shape}')
    embeddings = torch.tensor(embeddings)  # np.ndarray => torch.Tensor

    # train test split
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels, test_size=0.1, random_state=42
    )

    # Load data to Pytorch DataLoader
    train_dataloader, val_dataloader = create_data_loaders(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        train_labels=train_labels,
        val_labels=val_labels,
        batch_size=50
    )

    # check the device (GPU|CPU)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    # CNN-static: fastText pretrained word vectors are used and freezed during training.
    # fastText 事前学習された単語ベクトルが使われ、学習中は凍結される。
    set_seed(42)
    cnn_nlp, optimizer = initilize_model(
        pretrained_embedding=embeddings,
        freeze_embedding=True,
        learning_rate=0.25,
        dropout=0.5, device=device
    )

    cnn_nlp = train(model=cnn_nlp,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    epochs=20,
                    device=device
                    )

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


if __name__ == '__main__':
    os.chdir('text_cnn_test')
    main()
