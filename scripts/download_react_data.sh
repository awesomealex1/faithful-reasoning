#!/bin/bash

pip install gdown

mkdir -p .temp/

echo "\n\nDownloading raw HotpotQA data\n"
mkdir -p data/HotpotQA
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/HotpotQA/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/HotpotQA/hotpot_dev_distractor_v1.json

echo "\n\nDownloading raw 2WikiMultiHopQA data\n"
mkdir -p data/2WikiMultiHopQA
wget https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=0 -O .temp/2WikiMultiHopQA.zip
unzip -jo .temp/2WikiMultiHopQA.zip -d data/2WikiMultiHopQA -x "*.DS_Store"
rm data_ids.zip*

echo "\n\nDownloading hotpotqa wikipedia corpus (this will take ~5 mins)\n"
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -O .temp/wikpedia-paragraphs.tar.bz2
tar -xvf .temp/wikpedia-paragraphs.tar.bz2 -C data/HotpotQA
mv data/HotpotQA/enwiki-20171001-pages-meta-current-withlinks-abstracts data/HotpotQA/wikpedia-paragraphs

rm -rf .temp/