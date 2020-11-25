# Mean Teacher BERT discovers Fake News
Mean Teacher BERT discovers Fake News

## Abstract
Dissemination  of  fake  news  has  been  ubiquitous  with  theproliferation of social media platforms in this digital age. If the intentis  harmful,  fake  news  could  impact  negatively  on  our  society.  There-fore, it is vital to automatically detect the news potentially contain fakenews  before  reaching  to  large  audience.  Collecting  high  quality  of  la-beled corpus for the fake news detection models is a labor-intensive anda challenging step. To overcome this challenge, in this study, we evaluatesemi-supervised training schemas for fake news detection of news articleson publicly available datasets and compare the experimental results withthe state-of-art fake news detection models. The schemas that we eval-uate are Mean-Teacher, TODO ADD YOUR SIGNIFICANT RESULT.Our source code is publicly available1

## Datasets

For our paper, we use the following datasets:

* [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) contains news articles in politics and entertainment.
* [FakeHealth](https://zenodo.org/record/3862989) contains news articles in health.
* [NELA](https://dataverse.harvard.edu/dataverse/nela), [NELA-2017](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZCXSKG) and [NELA-2018](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ULHLCB) are used for unlabeled samples. From the [NELA-2019]() dataset, we use the aggregated labels. 

:warning: Please cite the papers of these studies if you use them. 

Run the [bash code](scripts/data_processing.sh) in order to get experiment folds.

You are supposed to see the processed files in `Data` folder
![ddata_directory](images/folder_dir.png)
## The Data\Input2 is the dataset on which overall comparision have been done and it is default dataset for model.The reports also generated from dataset2

## Starting
Python compiler is `3.7.9`.
Install libraries in requirements.txt:
```console
pip3 install -r requirements.txt
```

## Contributors
Ipek Baris

Bhupender kumar Saini

