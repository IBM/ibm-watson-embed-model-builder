# ibm-watson-embed-model-builder

This python library manages the process of building a collection of docker images that wrap individual watson_embedded
models for delivery with an embeddable watson runtime.

## Overview

Given a set of `watson_embed` models to be packaged, this tool can create a model manifest file describing the model
images and metadata to be created.

Given a model manifest file, this tool can fetch the model artifacts and package them into watson runtime compatible
model images along with useful metadata.

These operations are split into separate commands to facilitate easy parallelization in the CI system of your choice.

## Installation

```shell
pip install watson_embed_model_packager
```

## Usage

### 1. Building Manifests

To build a model manifest file from models hosted in an artifactory instance, you will need:

- a list of all module GUIDs to support
- a watson library and version
- an artifactory repo to search
- a target docker image repository for these model images to land
- artifactory credentials

```shell
export ARTIFACTORY_USERNAME=apikey
export ARTIFACTORY_API_KEY=my-artifactory-api-key
python3 -m watson_embed_model_packager setup \
    --module-guid 2cc95ffd-00fe-4d7d-9554-61d8777f3354 01b95845-c178-4d06-8598-0d49e23bd1a3 \
    --library-version watson_nlp:3.2.0 \
    --artifactory-repo https://my.artifactory.com/artifactory/my-watson-nlp-models/ \
    --target-registry my-docker-registry.com \
    --output-csv model-manifest.csv
```

To build a model manifest from model artifacts stored locally, you'll need a directory containing either unzipped
watson_embed models or valid .zip watson_embed model archives:

```shell
$ tree /path/to/models
/path/to/models
├── my_model_directory
│   ├── artifacts
│   │   ├── scheme.json
│   │   ├── word_sparse_vectors.npz
│   │   └── word_stats.pkl
│   └── config.yml
└── my_other_model.zip
```

Run the manifest setup with the `--local-model-dir` flag to create a manifest with these models:

```shell
python3 -m watson_embed_model_packager setup \
    --module-guid 2cc95ffd-00fe-4d7d-9554-61d8777f3354 01b95845-c178-4d06-8598-0d49e23bd1a3 \
    --library-version watson_nlp:3.2.0 \
    --target-registry my-docker-registry.com \
    --local-model-dir /path/to/models \
    --output-csv model-manifest.csv
```

### 2. Packaging model images

Use the model manifest to package your models into images with

```shell
python3 -m watson_embed_model_packager build --config model-manifest.csv
```

This requires `docker` installed and running on the system. Add `--push` to push the images after they are built.
