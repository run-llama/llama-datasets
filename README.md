# Llama Datasets 🦙📝

This repo is a companion repo to the [llama-hub repo](https://github.com/run-llama/llama-hub)
meant to be the actual storage of data files associated to a llama-dataset. Like
tools, loaders, and llama-packs, llama-datasets are offered through llama-hub. You
can view all of the available llama-hub artifacts conviently in the llama-hub
[website](https://llamahub.ai).

The primary use of a llama-dataset is for evaluating the performance of a RAG system.
In particular, it serves as a new test set (in traditional machine learning speak)
for one to build a RAG system over, predict on, and subsequently perform evaluations
comparing the predicted responses versus the reference responses.

## How to add a llama-dataset

Similar to the process of adding a tool / loader / llama-pack, adding a llama-
datset also requires forking the [llama-hub repo](https://github.com/run-llama/llama-hub)
and making a Pull Request. However, for a llama-dataset, only its metadata is checked into the llama-hub repo.
The actual dataset and it's source files are instead checked into this particular repo.
You will need to fork and clone that repo in addition to forking and cloning this one.

### Forking and cloning this repository

After forking this repo to your own Github account, the next step would be to
clone from your own fork. This repository is a LFS configured repo, and so, without
special care you may end up downloading large files to your local machine. As such,
we ask that when the time comes to clone your fork, please ensure that when you set the
environment variable `GIT_LFS_SKIP_SMUDGE` prior to calling the `git clone`
command:

```bash
# for bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:<your-github-user-name>/llama-datasets.git  # for ssh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/<your-github-user-name>/llama-datasets.git  # for https

# for windows its done in two commands
set GIT_LFS_SKIP_SMUDGE=1  
git clone git@github.com:<your-github-user-name>/llama-datasets.git  # for ssh

set GIT_LFS_SKIP_SMUDGE=1  
git clone https://github.com/<your-github-user-name>/llama-datasets.git  # for https
```

The high-level steps for adding a llama-dataset are as follows:

1. Create a `LabelledRagDataset` (the initial class of llama-dataset made available on llama-hub)
2. Generate a baseline result with a RAG system of your own choosing on the
`LabelledRagDataset`
3. Prepare the dataset's metadata (`card.json` and `README.md`)
4. Submit a Pull Request to this repo to check in the metadata
5. Submit a Pull Request to the [llama-datasets repository](https://github.com/run-llama/llama-datasets) to check in the `LabelledRagDataset` and the source files

To assist with the submission process, we have prepared a [submission template
notebook](https://github.com/run-llama/llama_index/blob/nerdai/add_template_nb/docs/examples/llama_dataset/ragdataset_submission_template.ipynb) that walks you through the above-listed steps. We highly recommend
that you use this template notebook.

## Usage Pattern

As mentioned earlier, llama-datasets are mainly used for evaluating RAG systems.
To perform the evaluation, the recommended usage pattern involves the application of the
`RagEvaluatorPack`. We recommend reading the [docs](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/root.html)
for the "Evaluation" module for more information.

```python
from llama_index.llama_dataset import download_llama_dataset
from llama_index.llama_pack import download_llama_pack
from llama_index import VectorStoreIndex

# download and install dependencies for benchmark dataset
rag_dataset, documents = download_llama_dataset(
  "PaulGrahamEssayDataset", "./data"
)

# build basic RAG system
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = VectorStoreIndex.as_query_engine()

# evaluate using the RagEvaluatorPack
RagEvaluatorPack = download_llama_pack(
  "RagEvaluatorPack", "./rag_evaluator_pack"
)
rag_evaluator_pack = RagEvaluatorPack(
    rag_dataset=rag_dataset,
    query_engine=query_engine
)
benchmark_df = rag_evaluate_pack.run()  # async arun() supported as well
```

Llama-datasets can also be downloaded directly using `llamaindex-cli`, which comes installed with the `llama-index` python package:

```bash
llamaindex-cli download-llamadataset PaulGrahamEssayDataset --download-dir ./data
```

After downloading them from `llamaindex-cli`, you can inspect the dataset and
it source files (stored in a directory `/source_files`) then load them into python:

```python
from llama_index import SimpleDirectoryReader
from llama_index.llama_dataset import LabelledRagDataset

rag_dataset = LabelledRagDataset.from_json("./data/rag_dataset.json")
documents = SimpleDirectoryReader(
    input_dir="./data/source_files"
).load_data()
```
