
# Main Script: Enhancing NLI Models

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from visualization import visualize_word_clouds, analyze_common_words
from creating_hypothesis import get_random_premise_and_label, generate_hypotheses
from filtering import analyze_incorrect_predictions, compute_and_display_dataset_similarities

def main():
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("pepa/roberta-base-snli")
    model = AutoModelForSequenceClassification.from_pretrained("pepa/roberta-base-snli")

    # Load the SNLI dataset
    dataset = load_dataset("snli")

    # Display dataset sample
    print(dataset)
    sample = dataset['train'][0]
    print("Sample from SNLI dataset:", sample)

    # Example usage of tokenizer and model
    inputs = tokenizer("A sample premise", "A sample hypothesis", return_tensors="pt")
    outputs = model(**inputs)
    print("BERT model output:", outputs)

    # Random premise and label
    premise, target_label = get_random_premise_and_label(dataset)
    print("Premise:", premise)
    print("Target label:", target_label)

    # Generate hypothesis
    hypothesis = generate_hypotheses(premise, target_label)
    print("Generated Hypothesis:", hypothesis)

    # Perform incorrect predictions analysis
    incorrect_predictions = analyze_incorrect_predictions(dataset, tokenizer, model)

    # Save incorrect predictions to CSV
    df_incorrect = pd.DataFrame(incorrect_predictions)
    df_incorrect.to_csv('incorrect.csv')

    # Visualizations and analysis
    visualize_word_clouds(df_incorrect, dataset)
    analyze_common_words(df_incorrect, dataset)

    # Compute dataset similarities
    compute_and_display_dataset_similarities(df_incorrect, dataset)

if __name__ == "__main__":
    main()
