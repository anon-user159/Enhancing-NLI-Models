
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def analyze_incorrect_predictions(dataset, tokenizer, model):
    incorrect_predictions = []
    for i in range(23800, 24500):
        example = dataset['train'][i]
        premise, target_label = example['premise'], example['label']
        if target_label not in [0, 1, 2]:
            continue
        hypothesis = f"Generated hypothesis for {premise}"  # Replace with model call
        inputs = tokenizer(premise, hypothesis, return_tensors="pt")
        outputs = model(**inputs)
        predicted_label = outputs.logits.argmax().item()
        if predicted_label != target_label:
            incorrect_predictions.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'original_label': target_label,
                'predicted_label': predicted_label
            })
    return incorrect_predictions

def compute_and_display_dataset_similarities(df_incorrect, dataset):
    datasets = {
        'train': pd.DataFrame(dataset['train'][:1000]),
        'generated': df_incorrect.head(1000)
    }

    similarities = {}
    tfidf = TfidfVectorizer()
    for (name1, data1), (name2, data2) in zip(datasets.items(), datasets.items()):
        if name1 != name2:
            all_texts = data1['premise'].tolist() + data2['premise'].tolist()
            tfidf_matrix = tfidf.fit_transform(all_texts)
            similarity = cosine_similarity(tfidf_matrix[:len(data1)], tfidf_matrix[len(data1):]).mean()
            similarities[f"{name1}-{name2}"] = similarity

    for pair, similarity in similarities.items():
        print(f"Similarity between {pair}: {similarity:.4f}")
