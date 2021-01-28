import argparse
import json
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


def process_fakes(path: str):
    data_path = Path(path)
    data = pd.read_csv(data_path, delimiter=",", encoding='latin1')
    data.rename(
        columns={"article_content": "content", "article_title": "title", "labels": "label",
                 "date": "publish_date", "unit_id": "news_id"},
        errors="raise", inplace=True)
    data["label"] = data.label.map(lambda x: "fake" if x == 0 else "true")
    data['publish_date'] = pd.to_datetime(data.publish_date)

    Path('Data/Processed').mkdir(parents=True, exist_ok=True)
    processed_dir = Path('Data/Processed') / 'FaKES.tsv'
    data.to_csv(processed_dir, sep='\t', index=False)
    logger.info(f"Processed data is saved to {processed_dir}")


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
    logger.info("Processing fakes")
    split_train_test(experimentfolds, 'fakes', 'Data/Processed/FaKES.tsv', ratio)
    logger.info("Processing fakehealth")
    split_train_test(experimentfolds, 'fakehealth', 'Data/Processed/FakeHealth.tsv', ratio)
    logger.info("Processing politifact")
    split_train_test(experimentfolds, 'politifact', 'Data/Processed/FakeNewsNet_Politifact.tsv', ratio)
    logger.info("Processing gossipcop")
    split_train_test(experimentfolds, 'gossipcop', 'Data/Processed/FakeNewsNet_Gossipcop.tsv', ratio)
    logger.info("Processing nela")
    split_train_test(experimentfolds, 'nela', 'Data/Processed/nela.tsv', ratio)
    logger.info("Processing covid")
    split_train_test(experimentfolds, 'covid', 'Data/Processed/covid.tsv', ratio)


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


def mask_labels(data, ratio_label):
    logger.info(data.groupby(["label"])["content"].count())
    if ratio_label == 1.0:
        return data
    data = data.sort_values(by='publish_date', ascending=False)
    labeled_len = round(len(data) * ratio_label)
    data.loc[:labeled_len, "label"] = -1

    assert data[data["publish_date"] == min(data.publish_date)]['label'].values[0] != -1 and \
           data[data["publish_date"] == max(data.publish_date)]['label'].values[0] == -1

    logger.info(data.groupby(["label"])["content"].count())
    return data


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

    reddit.rename(
        columns={"content": "text", "publishedDate": "publish_date"}, inplace=True)

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


def assign_label(source, ref_labels):
    if not ref_labels[ref_labels['source'] == source]['our_aggregation'].values:
        raise ValueError(f'Requesting source {source} not available in labels')
    label = ref_labels[ref_labels['source'] == source]['our_aggregation'].values[0]
    return label


def process_nela(nela_path):
    nela_dir = Path(nela_path)
    labels = pd.read_csv(nela_dir / 'labels.csv')
    reliable_sources = labels[labels['aggregated_label'] == 0.0]
    reliable_sources['our_aggregation'] = 'true'
    unreliable_sources = labels[labels['aggregated_label'] == 2.0]
    unreliable_sources['our_aggregation'] = 'fake'
    satire_sources = labels[labels['Media Bias / Fact Check, label'] == 'satire']
    satire_sources['our_aggregation'] = 'fake'
    new_labels = pd.concat([reliable_sources, unreliable_sources, satire_sources])

    nela_2017 = []
    for file_path in nela_dir.glob('**/*.txt'):
        with open(file_path) as f:
            json_txt = f.read()
            try:
                article_dict = json.loads(json_txt)
                normalized_source = normalize_source(article_dict["source"])
                label = assign_label(normalized_source, new_labels)
            except json.decoder.JSONDecodeError and ValueError:
                continue
            nela_2017.append({
                "title": article_dict["title"],
                "content": article_dict["content"],
                "source": normalized_source,
                "publish_date": str(file_path.absolute()).split("/")[-2],
                "label": label
            })

    nela_2017 = pd.DataFrame(nela_2017)
    nela_2017.dropna(subset=['publish_date'], inplace=True)  # remove nan values
    nela_2017.dropna(subset=['label'], inplace=True)  # remove nan values
    logger.info(f"Number of samples {len(nela_2017)}")
    processed_dir = Path('Data/Processed') / 'nela.tsv'
    nela_2017.to_csv(processed_dir, sep='\t', index=False)
    logger.info(f"NELA is saved to {processed_dir}")


def handle_date(date):
    date = pd.to_datetime(date, errors='coerce')
    if type(date) == pd.NaT:
        return None
    else:
        return f"{date.year}-{date.month}-{date.day}"


def process_covid(covid_path):
    data_dir = Path(covid_path)

    logger.info("Processing Recovery News Dataset")
    recovery_dir = data_dir / 'recovery-news-data.csv'
    recovery_data = pd.read_csv(recovery_dir, sep=',')
    recovery_data = recovery_data[['url', 'title', 'publisher', 'publish_date', 'body_text', 'reliability']]
    recovery_data.loc[recovery_data.reliability == 0, 'reliability'] = 'fake'
    recovery_data.loc[recovery_data.reliability == 1, 'reliability'] = 'true'
    recovery_data.rename(columns={"body_text": "content", "reliability": "label", "publisher": "source"}, inplace=True)
    recovery_data = recovery_data[['url', 'title', 'content', 'publish_date', 'label']]
    logger.info("Recovery News Stats")
    logger.info(recovery_data.groupby(['label'])['content'].count())

    logger.info("Processing CoAID Dataset")
    coaid_dir = data_dir / 'CoAID'

    coaid_all = []
    for filepath in coaid_dir.rglob("NewsFakeCOVID-19.csv"):
        data = pd.read_csv(filepath, sep=',')
        data = data[data['type'] == 'article']
        data['publish_date'] = data.publish_date.apply(
            lambda x: handle_date(x))
        data["label"] = "fake"
        data.rename(columns={"news_url": "url"},
                    inplace=True)
        coaid_all.append(data)

    for filepath in coaid_dir.rglob("NewsRealCOVID-19.csv"):
        data = pd.read_csv(filepath, sep=',')
        data = data[data['type'] == 'article']
        data['publish_date'] = data.publish_date.apply(
            lambda x: handle_date(x))
        data["label"] = "true"
        data.rename(columns={"news_url": "url"},
                    inplace=True)
        coaid_all.append(data)

    coaid_all = pd.concat(coaid_all)
    coaid_all = coaid_all[['url', 'title', 'content', 'publish_date', 'label']]
    logger.info("COAID Stats")
    logger.info(coaid_all.groupby(['label'])['content'].count())

    covid = pd.concat([recovery_data, coaid_all])
    covid.drop_duplicates(subset=['content'], inplace=True)

    Path('Data/Processed').mkdir(parents=True, exist_ok=True)
    processed_dir = Path('Data/Processed') / 'covid.tsv'
    covid.to_csv(processed_dir, sep='\t', index=False)
    logger.info(f"Processed data stats")
    logger.info(covid.groupby(['label'])['content'].count())
    logger.info(f"Processed data is saved to {processed_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fakehealth', type=str, help="Directory path of FakeHealth")
    parser.add_argument('--fakenewsnet', type=str, help="Directory path of FakeNewsNet")
    parser.add_argument('--fakes', type=str, help="Directory path of FakeNewsNet")
    parser.add_argument('--experimentfolds', type=str, help="Directory path of FakeNewsNet")
    parser.add_argument('--nela', type=str, help="Directory path of NELA datasets")
    parser.add_argument('--covid', type=str, help="Directory path of COVID datasets")
    parser.add_argument('--reddit', type=str, help="Directory path of Reddit dataset")
    parser.add_argument('--ratio', type=float, help="Ratio")
    args = parser.parse_args()

    if args.fakehealth:
        process_fakehealth(args.fakehealth)

    if args.fakenewsnet:
        process_fakenewsnet(args.fakenewsnet)

    if args.fakes:
        process_fakes(args.fakes)

    if args.experimentfolds:
        logger.info("Choosing experiment folder creation")
        create_experiment_data(args.experimentfolds, args.ratio)

    if args.nela:
        process_nela(args.nela)

    if args.covid:
        process_covid(args.covid)
