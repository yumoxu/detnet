# Weakly Supervised Domain Detection
This repository releases the code and data for weakly supervised domain detection. Please cite the following paper [[bib](https://www.mitpressjournals.org/action/showCitFormats?doi=10.1162/tacl_a_00287)] if you find our code /data resource useful to you,

> In this paper we introduce domain detection as a new natural language processing task. We argue that the ability to detect textual segments which are domain-heavy, i.e., sentences or phrases which are representative of and provide evidence for a given domain would enable the development of domain aware tools and increase the domain coverage for practical applications. We propose an encoder-detector framework for domain detection and bootstrap classifiers with multiple instance learning (MIL). The models are hierarchically organized and suited to multilabel classification. We demonstrate that despite learning from minimal supervision, our models can be applied to text spans of different granularities, languages, and genres.  We also explore the potential of domain detection for text summarization.

Should you have any query please contact me at [yumo.xu@ed.ac.uk](mailto:yumo.xu@ed.ac.uk).

## Project Structure

```bash
DomainDetection
│   README.md
│   spec-file.text
└───src
│   └───frame  # DetNet framework
│       │   encoder.py
│       │   detector.py
│       │   ...
│   └───config  # configuration files
│   └───data  # dataset parsing, building and piping
│   └───utils  # miscellaneous utils 
└───dataset
│   └───en  # English dataset
│       └───...
│   └───zh  # Chinese dataset
│       └───...
└───res  # resources (vocabulary)
│   └───vocab
│       └───en  # English vocabulary
│           │   vocab
│       └───zh  # Chinese vocabulary
│           │   vocab
└───model  # trained models
│   └───en  # English models
│       │   DetNet
│       │   ...
│   └───zh  # Chinese models
│       │   DetNet
│       │   ...
└───log

```

## Environment Setup

You can check the `spec-file.txt` provided in this project for the list of packages required. 

To create a suitable environment conviniently with `conda`, do:

```bash
conda create --name myenv --file spec-file.txt
```

or alternatively, you may prefer to install required packages into an existing environment:

```bash
conda install --name myenv --file spec-file.txt
```

## Dataset

You can download our datasets for both English and Chinese via [Google Drive](https://drive.google.com/drive/folders/1K5TdwoezGzzb19_2QjTuNipOX9kf1tUY?usp=sharing).

After uncompressing *.zip files, put them under `dataset/en` or `dataset/zh`, respectively. These include data for model training, development and test. Note that `test` is for document-level test, and `syn_docs`is for sentence-level test with synthesized contexts (check the algorithm proposed in our paper for details).

 `*.json` files include documents sampled from Wikipedia (in both `en` and `zh`) and NYT (in `en`); these documents are manually labeled via MTurk at both sentence-level and word-level for test purpose.