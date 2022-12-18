import pandas as pd

from src.config import MyConfig


def read_parquet_articles(parquet_path: str) -> pd.DataFrame:
    """parquetのパスを受け取ってpd.DataFrameとして返す.
    uciデータセットのformatに関しては以下linkを参照.
    https://www.kaggle.com/datasets/louislung/uci-news-aggregator-dataset-with-content?resource=download
    カラムは
    ['id', 'title', 'url', 'publisher', 'category', 'story', 'hostname',
       'timestamp', 'main_content', 'main_content_len']
    - id: article id
    - title:
    - url:
    - publisher:
    - category:
    - story:
    - hostname:
    - timestamp:
    - main_content:
    - main_content_len:
    """
    articles_df = pd.read_parquet(parquet_path)
    # articles_df = articles_df[articles_df.main_content.str.strip() != ""]
    # articles_df = articles_df[articles_df.main_content.notna()]

    # Add column based on title, ex) extract 食物設計 from 【食物設計（下）】
    if "story" not in articles_df.columns:
        articles_df["story"] = articles_df.title.str.extract("【(.*?)[（|】]")

    return articles_df


def get_valid_story_label(article_df: pd.DataFrame) -> pd.DataFrame:
    pass


def get_valid_category_label(article_df: pd.DataFrame) -> pd.DataFrame:
    pass


def main():
    article_df = read_parquet_articles(MyConfig.uci_news_data_path)
    print(article_df.columns)
    # article idの昇順でソート
    article_df = article_df.sort_values(by="id")
    print(article_df.head())

    # get valid story(テキストのより細かい分類ラベル)
    article_df = get_valid_story_label(article_df)

    # get valid category(テキストのよりラフな分類ラベル)
    article_df = get_valid_category_label(article_df)


if __name__ == "__main__":
    main()
