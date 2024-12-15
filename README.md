# Enhancing NLI Models

## Project Overview
Enhancing NLI Models is a project designed to improve Natural Language Inference (NLI) systems using the SNLI dataset and pre-trained RoBERTa models. It automates hypothesis generation, identifies incorrect predictions, filters data, and visualizes datasets with statistical insights.
![Workflow Diagram](flow_dia.png)
## Key Features
- **Hypothesis Generation**: Automatically generates hypotheses for premises and target labels.
- **Error Analysis**: Identifies and analyzes incorrect predictions.
- **Visualization**: Creates word clouds and analyzes common words.
- **Dataset Similarity**: Measures similarity between datasets using TF-IDF and cosine similarity.
## File Structure
- `enhancing_nli_models.py`: Main script to run the entire pipeline.
- `creating_hypothesis.py`: Contains functions to generate hypotheses.
- `filtering.py`: Includes logic for error filtering and similarity computation.
- `visualization.py`: Handles word clouds and text analysis visualizations.
- `phrasing.py`: Extracts key phrases and n-grams from text data.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/anon-user159/Enhancing-NLI-Models.git
   cd Enhancing-NLI-Models
   ```
2. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```

## Usage
Run the main script to execute the pipeline:
```bash
python enhancing_nli_models.py
```

## Dependencies
- `transformers`
- `datasets`
- `pandas`
- `matplotlib`
- `wordcloud`
- `scikit-learn`

## Example Output
- **Visualizations**: Word clouds for hypothesis text.
- **Processed CSV**: Incorrect predictions saved to `incorrect.csv`.


## Acknowledgements
- **SNLI Dataset**: Stanford Natural Language Inference dataset.
- **RoBERTa Model**: Pre-trained NLI model by Hugging Face.

- ## License
This project is licensed under the MIT License.



