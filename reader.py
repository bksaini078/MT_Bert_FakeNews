import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import dateutil.parser as dparser
import pandas as pd
from langdetect import detect

from logger import logger

THRESHOLD_SCORE = 3


def process_fakenewsnet(path: str):
    dir = Path(path)
    gossipcop = []
    politifact = []
    for filepath in dir.rglob("news content.json"):
        with open(filepath) as f:
            content = json.load(f)
            if len(content['text']) == 0:
                logger.debug(content['text'])
                logger.error('There is no content')
                continue
            if detect(content['text']) != 'en':
                continue
            data = {}
            data["content"] = content["text"]
            data["title"] = content["title"]
            data["publish_date"] = content["publish_date"]
            if not data["publish_date"]:
                continue
            else:
                data["publish_date"] = datetime.fromtimestamp(data["publish_date"]).strftime("%Y-%m-%d")

            data["url"] = content["url"]
            filepath = str(filepath)
            data["news_id"] = filepath
            data["label"] = "fake" if "fake" in filepath else "true"

            if "gossipcop" in filepath:
                gossipcop.append(data)
            else:
                politifact.append(data)

    logger.info("Stats of Gossipcop")
    gossipcop = pd.DataFrame(gossipcop)
    gossipcop.dropna(subset=['publish_date'], inplace=True)  # remove nan values
    logger.info(gossipcop.groupby(["label"])["url"].count())

    logger.info("Stats of Politifact")
    politifact = pd.DataFrame(politifact)
    politifact.dropna(subset=['publish_date'], inplace=True)  # remove nan values
    logger.info(politifact.groupby(["label"])["url"].count())

    Path('Data/Processed').mkdir(parents=True, exist_ok=True)
    processed_dir = Path('Data/Processed') / 'FakeNewsNet_Gossipcop.tsv'
    gossipcop.to_csv(processed_dir, sep='\t', index=False)

    processed_dir = Path('Data/Processed') / 'FakeNewsNet_Politifact.tsv'
    politifact.to_csv(processed_dir, sep='\t', index=False)


def process_fakehealth(path: str):
    dir = Path(path)
    reviews = pd.read_json(dir / 'reviews/HealthStory.json')
    fake_stories = reviews[reviews["rating"] < THRESHOLD_SCORE]
    true_stories = reviews[reviews["rating"] >= THRESHOLD_SCORE]

    logger.info("Stats of Health Story")
    logger.info(f"Num of fake stories {len(fake_stories)}")
    logger.info(f"Num of true stories {len(true_stories)}")
    logger.info(f"Num of unique sources in fake stories {len(fake_stories.news_source.unique())}")
    logger.info(f"Num of unique sources in true stories {len(true_stories.news_source.unique())}")
    logger.info(
        f"Num of overlapped sources {len(set(fake_stories.news_source.unique()) - set(true_stories.news_source.unique()))}")

    filtered_fake_stories = pd.DataFrame(extract_content(dir, "HealthStory", fake_stories.news_id.values, "fake"))
    filtered_true_stories = pd.DataFrame(extract_content(dir, "HealthStory", true_stories.news_id.values, "true"))

    releases = pd.read_json(dir / 'reviews/HealthRelease.json')
    fake_releases = releases[releases["rating"] < THRESHOLD_SCORE]
    true_releases = releases[releases["rating"] >= THRESHOLD_SCORE]

    logger.info("Stats of Health Releases")
    logger.info(f"Num of fake releases {len(fake_stories)}")
    logger.info(f"Num of true releases {len(true_stories)}")
    logger.info(f"Num of unique sources in fake releases {len(fake_releases.news_source.unique())}")
    logger.info(f"Num of unique sources in true releases {len(true_releases.news_source.unique())}")
    logger.info(
        f"Num of overlapped sources {len(set(fake_stories.news_source.unique()) - set(true_stories.news_source.unique()))}")

    filtered_fake_releases = pd.DataFrame(extract_content(dir, "HealthRelease", fake_releases.news_id.values, "fake"))
    filtered_true_releases = pd.DataFrame(extract_content(dir, "HealthRelease", true_releases.news_id.values, "true"))
    merged_data = pd.concat(
        [filtered_fake_releases, filtered_true_stories, filtered_fake_stories, filtered_true_releases])

    Path('Data/Processed').mkdir(parents=True, exist_ok=True)
    processed_dir = Path('Data/Processed') / 'FakeHealth.tsv'
    merged_data.to_csv(processed_dir, sep='\t', index=False)
    logger.info(f"Merged data is saved to {processed_dir}")


def extract_content(dir, news_source, news_ids, label):
    processed_data = []
    for news_id in news_ids:
        data = {}
        fname = f"content/{news_source}/{news_id}.json"
        if not (dir / fname).exists():
            logger.error(f"{news_id} does not exist")
            continue
        with open(dir / fname) as f:
            news_item = json.load(f)
            data["content"] = news_item["text"]
            data["title"] = news_item["title"]
            if news_item["publish_date"]:
                data["publish_date"] = datetime.fromtimestamp(news_item["publish_date"])
            elif "lastPublishedDate" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["lastPublishedDate"]
            elif "dcterms.modified" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["dcterms.modified"]
            elif "article.updated" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["article.updated"]
            elif "date" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["date"]
            else:
                try:
                    data["publish_date"] = dparser.parse(news_item["text"][:100], fuzzy=True)
                except:
                    logger.error(f"{news_id} does not have a publish_date")
                    continue
            data["url"] = news_item["url"]
            data["news_id"] = news_id
            data["label"] = label
            processed_data.append(data)
    logger.info(f"Num of processed {label} news with date {len(processed_data)}")
    return processed_data


def create_experiment_data(experimentfolds, ratio):
    '''
    Sort based on published date, seperate each dataset into 80% of training, 20% testing
    Apply 5-fold cross validation sets from training set.
    '''
    experimentfolds = Path(experimentfolds)
    experimentfolds.mkdir(parents=True, exist_ok=True)
    logger.info("Processing fakehealth")
    split_train_test(experimentfolds, 'fakehealth', 'Data/Processed/FakeHealth.tsv', ratio)
    logger.info("Processing politifact")
    split_train_test(experimentfolds, 'politifact', 'Data/Processed/FakeNewsNet_Politifact.tsv', ratio)
    logger.info("Processing gossipcop")
    split_train_test(experimentfolds, 'gossipcop', 'Data/Processed/FakeNewsNet_Gossipcop.tsv', ratio)


def split_train_test(experimentfolds, experiment_folder, datapath, ratio):
    data = pd.read_csv(datapath, sep="\t")
    data = data.sort_values(by='publish_date', ascending=True)
    test_index = round(len(data) * ratio) + 1
    train_index = len(data) - test_index
    output_path = experimentfolds / experiment_folder
    test = data[train_index + 1:]
    train = data[:train_index + 1]

    assert len(train) + len(test) == len(data)

    output_path.mkdir(parents=True, exist_ok=True)
    test.to_csv(output_path / 'test.tsv', sep="\t", index=False)
    train.to_csv(output_path / 'train.tsv', sep="\t", index=False)

    # kf = KFold(n_splits=5, random_state=42, shuffle=True)
    # kf.get_n_splits(train)
    #
    # unlabeled_all = pd.read_csv("Data/Processed/Reddit_All.tsv", sep="\t")
    # # unlabeled_mix = pd.read_csv("Data/Processed/Nela_Mix.tsv", sep="\t")
    #
    # date_threshold = min(test["publish_date"])
    # logger.info(date_threshold)
    # _unlabeled_all = unlabeled_all[unlabeled_all["publish_date"] < date_threshold]
    # # _unlabeled_mix = unlabeled_mix[unlabeled_mix["publish_date"] < date_threshold]
    #
    # logger.info(max(_unlabeled_all["publish_date"]))
    # assert max(_unlabeled_all["publish_date"]) < date_threshold
    # # assert max(_unlabeled_mix["publish_date"]) < date_threshold
    #
    # train_remove = train[train["publish_date"] == date_threshold]
    # train = train[train["publish_date"] < date_threshold]
    #
    # test = pd.concat([train_remove, test])
    #
    # assert max(train["publish_date"]) < date_threshold
    #
    # _unlabeled_all.to_csv(output_path / 'unlabeled_all.tsv', sep="\t", index=False)
    # # _unlabeled_mix.to_csv(output_path / 'unlabeled_mix.tsv', sep="\t", index=False)
    # logger.info(f"Unlabeled data samples {len(_unlabeled_all)}")
    # train_fname = 'train_{}.tsv'
    # dev_fname = 'dev_{}.tsv'
    # train.reset_index(inplace=True)
    # print(f"Test len {len(test)}")
    # print(test.groupby(["label"])["content"].count())
    # for idx, (train_index, test_index) in enumerate(kf.split(train.label)):
    #     print(f"Fold {idx}")
    #     _train, _dev = train.loc[train_index], train.loc[test_index]
    #     print(f"Train len {len(_train)}")
    #     print(_train.groupby(["label"])["content"].count())
    #     print(f"Dev len {len(_dev)}")
    #     print(_dev.groupby(["label"])["content"].count())
    #     _train.to_csv(output_path / train_fname.format(idx), sep="\t", index=False)
    #     _train.to_csv(output_path / train_fname.format(idx), sep="\t", index=False)
    #     _dev.to_csv(output_path / dev_fname.format(idx), sep="\t", index=False)


def normalize_source(source):
    source = source.replace(' ', '')
    source = source.lower()
    if '.' in source:
        source = source.split(".")[0]
    return source


def process_reddit_news(reddit_path, nela_path):
    reddit_path = Path(reddit_path)
    nela_dir = Path(nela_path)
    labels = pd.read_csv(nela_dir / 'labels.csv')
    reliable_sources = labels[labels['aggregated_label'] == 0.0]['source'].unique()
    unreliable_sources = labels[labels['aggregated_label'] == 2.0]['source'].unique()
    satire_sources = labels[labels['Media Bias / Fact Check, label'] == 'satire']['source'].unique()

    reddit = pd.read_csv(reddit_path, sep="\t")
    reddit["source"] = reddit.source.map(lambda x: normalize_source(x))

    print(reddit.source.unique())
    reddit.rename(
        columns={"content": "text", "publishedDate": "publish_date"},
        errors="raise", inplace=True)

    reliable = reddit[reddit.source.isin(reliable_sources)]
    unreliable = reddit[reddit.source.isin(unreliable_sources)]
    satire = reddit[reddit.source.isin(satire_sources)]
    reliable["label"] = "reliable"
    unreliable["label"] = "unreliable"
    satire["label"] = "satire"

    logger.info(f"Reliable news in reddit {len(reliable)}")
    logger.info(f"Unreliable news in reddit {len(unreliable)}")
    logger.info(f"Satire news in reddit {len(satire)}")

    reddit_all = pd.concat([reliable, unreliable, satire])

    logger.info(f"Number of samples {len(reddit_all)}")
    processed_dir = Path('Data/Processed') / 'Reddit_all.tsv'
    reddit_all.to_csv(processed_dir, sep='\t', index=False)
    logger.info(f"Reddit data is saved to {processed_dir}")


def process_nela(nela_path):
    nela_dir = Path(nela_path)
    labels = pd.read_csv(nela_dir / 'labels.csv')
    reliable_sources = labels[labels['aggregated_label'] == 0.0]['source'].unique()
    unreliable_sources = labels[labels['aggregated_label'] == 2.0]['source'].unique()
    satire_sources = labels[labels['Media Bias / Fact Check, label'] == 'satire']['source'].unique()

    nela_2018_dir = nela_dir / 'NELA-2018/articles.db'
    nela_2018_cnx = sqlite3.connect(nela_2018_dir)
    nela_2018 = pd.read_sql_query("SELECT * FROM articles", nela_2018_cnx)

    nela_2018["normalized_source"] = nela_2018.source.map(lambda x: normalize_source(x))
    nela_2018 = nela_2018[["normalized_source", "name", "content", "date"]]
    nela_2018.rename(
        columns={"normalized_source": "source", "name": "title", "content": "text", "date": "publish_date"},
        errors="raise", inplace=True)

    reliable_nela_2018 = nela_2018[nela_2018.source.isin(reliable_sources)]
    unreliable_nela_2018 = nela_2018[nela_2018.source.isin(unreliable_sources)]
    satire_nela_2018 = nela_2018[nela_2018.source.isin(satire_sources)]

    logger.info(f"Reliable news in NELA 2018 {len(reliable_nela_2018)}")
    logger.info(f"Unreliable news in NELA 2018 {len(unreliable_nela_2018)}")
    logger.info(f"Satire news in NELA 2018 {len(satire_nela_2018)}")

    nela_2017_dir = nela_dir / 'NELA-2017'
    nela_2017 = []
    for file_path in nela_2017_dir.glob('**/*.txt'):
        with open(file_path) as f:
            json_txt = f.read()
            try:
                article_dict = json.loads(json_txt)
            except json.decoder.JSONDecodeError:
                continue

            normalized_source = normalize_source(article_dict["source"])
            nela_2017.append({
                "title": article_dict["title"],
                "text": article_dict["content"],
                "source": normalized_source,
                "publish_date": str(file_path.absolute()).split("/")[10]
            })

    nela_2017 = pd.DataFrame(nela_2017)
    logger.info(f"Before NELA 2017 {len(nela_2017)}")
    nela_2017.dropna(subset=['publish_date'], inplace=True)  # remove nan values
    logger.info(f"After removing articles without dates {len(nela_2017)}")

    logger.info(f"Before NELA 2018 {len(nela_2018)}")
    nela_2018.dropna(subset=['publish_date'], inplace=True)  # remove nan values
    logger.info(f"After removing articles without dates {len(nela_2018)}")

    reliable_nela_2017 = nela_2017[nela_2017.source.isin(reliable_sources)]
    reliable_nela_2017["label"] = "reliable"
    unreliable_nela_2017 = nela_2017[nela_2017.source.isin(unreliable_sources)]
    unreliable_nela_2017["label"] = "unreliable"
    satire_nela_2017 = nela_2017[nela_2017.source.isin(satire_sources)]
    satire_nela_2017["label"] = "satire"
    nela_all = pd.concat([nela_2017, nela_2018])
    logger.info(f"Number of samples {len(nela_all)}")
    Path('Data/Processed').mkdir(parents=True, exist_ok=True)
    processed_dir = Path('Data/Processed') / 'Nela_All.tsv'
    nela_all.to_csv(processed_dir, sep='\t', index=False)
    logger.info(f"NELA all is saved to {processed_dir}")

    nela_mix = pd.concat(
        [reliable_nela_2017, reliable_nela_2018, unreliable_nela_2017, unreliable_nela_2018, satire_nela_2017,
         satire_nela_2018])
    logger.info(f"Number of samples {len(nela_mix)}")
    processed_dir = Path('Data/Processed') / 'Nela_Mix.tsv'
    nela_mix.to_csv(processed_dir, sep='\t', index=False)
    logger.info(f"NELA mix is saved to {processed_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fakehealth', type=str, help="Directory path of FakeHealth")
    parser.add_argument('--fakenewsnet', type=str, help="Directory path of FakeNewsNet")
    parser.add_argument('--experimentfolds', type=str, help="Directory path of FakeNewsNet")
    parser.add_argument('--nela', type=str, help="Directory path of NELA datasets")
    parser.add_argument('--reddit', type=str, help="Directory path of Reddit dataset")
    parser.add_argument('--ratio', type=float, help="Ratio")
    args = parser.parse_args()

    if args.fakehealth:
        process_fakehealth(args.fakehealth)

    if args.fakenewsnet:
        process_fakenewsnet(args.fakenewsnet)

    if args.experimentfolds:
        logger.info("Choosing experiment folder creation")
        create_experiment_data(args.experimentfolds, args.ratio)

    if args.nela:
        logger.info("Choosing processing NELA")
        # process_nela(args.nela)
        process_reddit_news(args.reddit, args.nela)
