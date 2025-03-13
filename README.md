# Reddit NER Pipeline with roBERTa

This is an enhanced Named Entity Recognition (NER) pipeline for extracting entities from Reddit posts stored in JSONL files. The pipeline uses the Hugging Face roBERTa model fine-tuned on WNUT2017 to extract entities from the text of Reddit posts and creates a JSON file connecting entities to the posts they appear in.

## Features

- Recursively processes JSONL files in a directory structure
- Extracts named entities using the roBERTa model fine-tuned on WNUT2017
- Creates a JSON file connecting entities to posts for easy analysis
- Provides a search functionality to find posts mentioning specific entities
- Supports filtering by entity type
- Shows top entities by frequency
- Includes a Jupyter notebook for analyzing the entity data

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r roberta_requirements.txt
```

## Usage

### Running the Pipeline

To process all JSONL files and create the entity index:

```bash
python roberta_ner_pipeline.py --data_dir 2025 --output_dir roberta_ner_output
```

Options:
- `--data_dir`: Directory containing the JSONL files (default: '2025')
- `--output_dir`: Directory to save the output files (default: 'roberta_ner_output')
- `--model`: Hugging Face model to use (default: 'mbastardi24/roBERTa-finetuned-wnut2017')

### Searching for Entities

To search for posts mentioning a specific entity:

```bash
python roberta_ner_pipeline.py --search --query "Donald Trump"
```

Options:
- `--search`: Run in search mode
- `--query`: Entity to search for
- `--entity_type`: Filter by entity type (B-person, B-location, etc.)
- `--top_entities`: Show top entities by frequency

### Examples

Show top entities:

```bash
python roberta_ner_pipeline.py --search --top_entities
```

Show top people:

```bash
python roberta_ner_pipeline.py --search --top_entities --entity_type B-person
```

Search for posts mentioning Donald Trump:

```bash
python roberta_ner_pipeline.py --search --query "Donald Trump"
```

Search for posts mentioning Donald Trump, filtered to only include entities recognized as people:

```bash
python roberta_ner_pipeline.py --search --query "Donald Trump" --entity_type B-person
```

## Entity Types

The roBERTa model fine-tuned on WNUT2017 extracts the following entity types:

- B-corporation: Beginning of a corporation entity
- I-corporation: Inside of a corporation entity
- B-creative-work: Beginning of a creative work entity
- I-creative-work: Inside of a creative work entity
- B-group: Beginning of a group entity
- I-group: Inside of a group entity
- B-location: Beginning of a location entity
- I-location: Inside of a location entity
- B-person: Beginning of a person entity
- I-person: Inside of a person entity
- B-product: Beginning of a product entity
- I-product: Inside of a product entity

The "B-" prefix indicates the beginning of an entity, while the "I-" prefix indicates the inside (continuation) of an entity.

## Data Format

The pipeline saves the entity data in a JSON file with the following structure:

```json
{
  "Entity Name": {
    "types": ["B-person", "I-person"],
    "post_count": 5,
    "posts": [
      {
        "id": "post_id",
        "title": "Post Title",
        "text": "Post Text",
        "permalink": "Post Permalink",
        "score": 10,
        "created_utc": "1739559520.0"
      },
      ...
    ]
  },
  ...
}
```

This format makes it easy to analyze the data and find connections between entities and posts.

## Analysis Notebook

The repository includes a Jupyter notebook (`entity_analysis.ipynb`) for analyzing the entity data. The notebook demonstrates how to:

- Load and explore the entity data
- Find the most frequently mentioned entities
- Analyze entity co-occurrences
- Search for specific entities
- Analyze entity mentions over time
- Export entity data to CSV files for further analysis

## Performance Considerations

- The pipeline uses the Hugging Face transformers library to load and run the roBERTa model
- Processing large amounts of text with the transformer model can be computationally intensive
- For better performance, consider running the pipeline on a machine with a GPU 

# Reddit NER Pipeline

This is a Named Entity Recognition (NER) pipeline for extracting entities from Reddit posts stored in JSONL files. The pipeline extracts entities such as people, organizations, locations, and more from the text of Reddit posts and creates an index connecting entities to the posts they appear in.

## Features

- Recursively processes JSONL files in a directory structure
- Extracts named entities using spaCy's NER model
- Creates an index connecting entities to posts
- Provides a search functionality to find posts mentioning specific entities
- Supports filtering by entity type (PERSON, ORG, GPE, etc.)
- Shows top entities by frequency

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the spaCy model:

```bash
python -m spacy download en_core_web_lg
```

## Usage

### Running the Pipeline

To process all JSONL files and create the entity index:

```bash
python ner_pipeline.py --data_dir 2025 --output_dir ner_output
```

Options:
- `--data_dir`: Directory containing the JSONL files (default: '2025')
- `--output_dir`: Directory to save the output files (default: 'ner_output')
- `--model`: spaCy model to use (default: 'en_core_web_lg')

### Searching for Entities

To search for posts mentioning a specific entity:

```bash
python ner_pipeline.py --search --query "Donald Trump"
```

Options:
- `--search`: Run in search mode
- `--query`: Entity to search for
- `--entity_type`: Filter by entity type (PERSON, ORG, GPE, etc.)
- `--top_entities`: Show top entities by frequency

### Examples

Show top entities:

```bash
python ner_pipeline.py --search --top_entities
```

Show top people:

```bash
python ner_pipeline.py --search --top_entities --entity_type PERSON
```

Search for posts mentioning Donald Trump:

```bash
python ner_pipeline.py --search --query "Donald Trump"
```

Search for posts mentioning Donald Trump, filtered to only include entities recognized as people:

```bash
python ner_pipeline.py --search --query "Donald Trump" --entity_type PERSON
```

## Entity Types

The pipeline extracts the following entity types:

- PERSON: People, including fictional characters
- ORG: Companies, agencies, institutions, etc.
- GPE: Countries, cities, states
- LOC: Non-GPE locations, mountain ranges, bodies of water
- PRODUCT: Objects, vehicles, foods, etc.
- EVENT: Named hurricanes, battles, wars, sports events, etc.
- WORK_OF_ART: Titles of books, songs, etc.
- FAC: Buildings, airports, highways, bridges, etc.

## Performance Considerations

- The pipeline uses spaCy's `en_core_web_lg` model by default, which provides a good balance between accuracy and performance.
- For better accuracy but slower processing, you can use `en_core_web_trf` (transformer-based model).
- For faster processing but lower accuracy, you can use `en_core_web_sm` or `en_core_web_md`.
- The pipeline disables unnecessary components in the spaCy pipeline for better performance.
- The entity index and post data are saved as pickle files for efficient storage and loading. 

