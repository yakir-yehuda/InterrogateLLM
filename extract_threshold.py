import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import util
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve

from language_models import SBert, E5, GPTEmbedding, BERTembedding



emebedding_models = {'sbert': SBert(), 
                     'e5': E5(), 
                     'ada002': GPTEmbedding(),
                     'bert': BERTembedding()
                    }


def evaluate_threshold(gt, cosine_scores):

    # Option 1: ROC Curve Analysis
    fpr, tpr, thresholds_roc = roc_curve(gt, cosine_scores)
    best_threshold_roc = thresholds_roc[np.argmax(tpr - fpr)]
    y_pred_roc = cosine_scores > best_threshold_roc
    best_accuracy_roc = accuracy_score(gt, y_pred_roc)

    # Option 2: Precision-Recall Curve Analysis
    precision, recall, thresholds_pr = precision_recall_curve(gt, cosine_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold_pr = thresholds_pr[np.argmax(f1_scores)]
    y_pred_pr = cosine_scores > best_threshold_pr
    best_accuracy_pr = accuracy_score(gt, y_pred_pr)

    # Option 3: F1 Score Maximization
    best_threshold_f1 = thresholds_pr[np.argmax(f1_scores)]
    y_pred_f1 = cosine_scores > best_threshold_f1
    best_accuracy_f1 = accuracy_score(gt, y_pred_f1)

    # Print results
    print("\nOption 1 - ROC Curve Analysis:")
    print(f"Best Threshold: {best_threshold_roc}")
    print(f"Validation Accuracy: {best_accuracy_roc}")

    print("\nOption 2 - Precision-Recall Curve Analysis:")
    print(f"Best Threshold: {best_threshold_pr}")
    print(f"Validation Accuracy: {best_accuracy_pr}")

    print("\nOption 3 - F1 Score Maximization:")
    print(f"Best Threshold: {best_threshold_f1}")
    print(f"Validation Accuracy: {best_accuracy_f1}")


def create_sentences_pairs_emebeddings(embedding_model, samples):
    dataset = load_dataset("embedding-data/QQP_triplets")
    
    gt = []
    cosine_scores = []
    
    for i, data in tqdm(enumerate(dataset['train'])):
        if i > samples:
            break
        data = data['set']
        query = data['query']
        pos_pair = data['pos'][0]
        neg_pair = data['neg'][0]
        
        
        query_embedding = embedding_model.submit_embedding_request(query)
        pos_pair_embedding = embedding_model.submit_embedding_request(pos_pair)
        neg_pair_embedding = embedding_model.submit_embedding_request(neg_pair)
        
        pos_cos_sim = util.cos_sim(query_embedding, pos_pair_embedding).item()
        neg_cos_sim = util.cos_sim(query_embedding, neg_pair_embedding).item()
        gt += [1, 0]
        cosine_scores += [pos_cos_sim, neg_cos_sim]
    
    gt = np.array(gt)
    cosine_scores = np.array(cosine_scores)
    return gt, cosine_scores
     
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_model_name', type=str, default='bert', choices=['ada002', 'sbert', 'e5', 'bert'],
                        help='embedding model name')
    parser.add_argument('--samples', type=int, default=10000,
                        help='dataset samples to use for threshold calculation')
    args = parser.parse_args()
    
    print(f'emebdding model: {args.embedding_model_name}')
    # use the selected embedding model to calculate the cosine similarity between the sentences
    emebedding_model = emebedding_models[args.embedding_model_name]
    # create cosine similarity scores and ground truth labels for the sentences pairs
    gt, cosine_scores = create_sentences_pairs_emebeddings(emebedding_model, samples=args.samples)
    # calculate the optimal threshold for the cosine similarity scores using the ground truth labels (according to the roc curve and the tpr-fpr difference)
    evaluate_threshold(gt, cosine_scores)
