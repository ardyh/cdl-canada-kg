# ComplexData

We have not added the underlying data to this repo for two reasons:
1. The datasets are large. We work with a filtered subset (by time, topic, and social media impressions).
2. The datasets belong to a larger project we are participating in with a research group and we do not have approval to publish them yet.

## How to read/run

Our pipeline is broken down into the following steps, assuming the data is already available:
1. Data filtering
2. Extraction
3. Deduplication
4. KG ontology/triple creation

### 1. Data filtering
We use the `filtering_clean.ipynb` notebook to filter data (by keywords and usernames) for specific topics and political figures for a temporal subset of the larger dataset.

### 2. Extraction
The `extraction_x_bluesky.py` script performs extraction of our entities from social media posts, using GPT-4o-mini.

We also performed an evaluation of our extraction quality against a manually annotated subset of the data, which is found in the `extraction_development.ipynb` notebook.

### 3. Deduplication
We use notebook `deduplication.ipynb` to prototype our deduplication method (agglomerative clustering of extractions' embeddings with [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)).

The actual deduplication is run with `deduplication_best.py`. Other relevant scripts are `deduplication.py` and `deduplication_kmeans.py` (where we search over different hyperparameters for agglomerative and k-means clustering, respectively, to decide on the best setting).

### 4. KG ontology/triple creation
We use the notebook `kg_creation.ipynb` to define the ontology and add triples to our Neo4J Aura instance. This notebook loads the social media data, extractions, and deduplications created previously, defines a KG structure (in the class `SocialMediaKnowledgeGraph`), and formats/adds our data to that structure.
