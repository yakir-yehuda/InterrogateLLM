import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
from language_models import SBert, GPTEmbedding, E5, BERTembedding
from sentence_transformers import util
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score , precision_recall_curve

import utils

embedding_models = {'sbert': SBert(),
                    'e5': E5(),
                    'ada002': GPTEmbedding(),
                    'bert': BERTembedding()
                    }
# this threshold extracted from QQP dataset for each embedding model
embedding_model_thresholds = {'sbert': 0.782,
                              'e5': 0.896,
                              'ada002': 0.915,
                              'bert': 0.853
                              }

heuristic_thresholds = {'movies':0.8, 'books':2, 'world':None}

def movies_answer_heuristic(predicted_answer, gt_answer, threshold=0.8):
    predicted_cast = utils.spacy_extract_entities(predicted_answer)
    intersection, union = utils.calculate_intersection_and_union(gt_answer['movie_cast'], predicted_cast)

    answer_simple_heuristic = len(intersection) / len(predicted_cast) > threshold if len(predicted_cast) != 0 else True
    return answer_simple_heuristic

def books_answer_heuristic(predicted_answer, gt_answer, threshold=2):
    answer_simple_heuristic = sum([1 for x in gt_answer.values() if utils.check_entity_in_sentence(x, predicted_answer)])

    return answer_simple_heuristic >= threshold

def world_answer_heuristic(predicted_answer, gt_answer):
    for x in gt_answer.values():
        for i in x:
            if utils.check_entity_in_sentence(i, predicted_answer):
                return True
    return False


def calc_ans_heuristic(predicted_answer, gt_answer, dataset_name, heuristic_threshold):
    if dataset_name == 'books':
        gt = [books_answer_heuristic(x, y, heuristic_threshold) for x, y in zip(predicted_answer, gt_answer)]
    elif dataset_name == 'movies':
        gt = [movies_answer_heuristic(x, y, heuristic_threshold) for x, y in zip(predicted_answer, gt_answer)]
    elif dataset_name == 'world':
        gt = [world_answer_heuristic(x, y) for x, y in zip(predicted_answer, gt_answer)]
    else:
        gt = []

    return gt


def auc_plot(gt, pred, title, file_name, save_path='./'):
    pred = [x if x <= 1 else 1.0 for x in pred]
    gt = np.array(gt)
    gt = 1 - gt
    
    pred = 1 - np.array(pred)
    
    # calculate roc curve
    fpr, tpr, _ = roc_curve(gt, pred)


    ns_probs = [0 for _ in range(len(gt))]
    ns_fpr, ns_tpr, _ = roc_curve(gt, ns_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(fpr, tpr, marker='.', label='Model')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    plt.text(0.1, 0.02, f'AUC: {roc_auc_score(gt, pred):.3f}\n', fontsize=8, bbox=props)

    # title
    plt.title(title)

    # show the plot
    save_path = os.path.join(save_path, file_name)
    plt.savefig(save_path)
    plt.cla()



def calc_metrics(gt, pred, optimal_threshold):
    pred = [x if x <= 1 else 1.0 for x in pred]
    gt = np.array(gt)
    gt = 1 - gt
    
    bal_acc = balanced_accuracy_score(gt, np.array(pred) < optimal_threshold)

    acc = accuracy_score(gt, np.array(pred) < optimal_threshold)
    p_score = precision_score(gt, np.array(pred) < optimal_threshold)
    r_score = recall_score(gt, np.array(pred) < optimal_threshold)
    f1 = f1_score(gt, np.array(pred) < optimal_threshold)
    auc = roc_auc_score(gt, 1 - np.array(pred))
    
    print(f'Auc: {auc:.3f}, Bal Acc: {bal_acc:.3f}, Acc: {acc:.3f}, F1: {f1:.3f}')
    print(f'Hallucination rate: {np.mean(gt):.3f}')
    return acc, bal_acc, p_score, r_score, f1

def calc_auc_and_acc(base_dir, heuristic_threshold, dataset_name='books', embedding_model=None, threshold=0.0):
    current_dir = os.path.join(base_dir, 'res_pkl')
    for f in os.listdir(current_dir):
        sample_sizes = [int(i.split('_')[-1].split('.')[0]) for i in os.listdir(current_dir)]
        sample_size = int(f.split('_')[-1].split('.')[0])
        file_path = os.path.join(current_dir, f)
        max_sample = 3000 if dataset_name != 'world' else max(sample_sizes)
        if sample_size != max_sample:
            continue
        print (f'File path: {file_path}')    
        with open(file_path, 'rb') as handle:
            results = pkl.load(handle)
            
            gt_answers = [res['original_answer'] for res in results]
            pred_ans = [res['predicted_answer'] for res in results]
            gt = calc_ans_heuristic(pred_ans, gt_answers, dataset_name, heuristic_threshold)
             
            pred_questions_cosine = []
            for i, res in tqdm(enumerate(results)):
                original_question = res['original_question']
                predicted_answer = res['predicted_answer']
                question_embedding = embedding_model.submit_embedding_request(original_question)
                answer_embedding = embedding_model.submit_embedding_request(predicted_answer)
                cos_sim = util.cos_sim(question_embedding, answer_embedding).item()
                pred_questions_cosine.append(cos_sim)

            calc_metrics(gt, pred_questions_cosine, threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='world', choices=['books', 'movies', 'world'],
                        help='dataset name')

    parser.add_argument('--ans_model', type=str, default='gpt', choices=['gpt', 'llamaV2-7', 'llamaV2-13'],
                        help='llm model name')

    parser.add_argument('--embedding_model_name', type=str, default='sbert', choices=['ada002', 'sbert', 'e5','bert'],
                        help='embedding model name')
    args = parser.parse_args()

    question_models_name = ['gpt', 'llamaV2-7', 'llamaV2-13']
    exp_dir = f'{args.dataset_name}_experiments'
    
    exp_dir = os.path.join('.', exp_dir, args.ans_model, '-'.join(question_models_name), 'ada002')

    embedding_model = embedding_models[args.embedding_model_name]
    embedding_model_threshold = embedding_model_thresholds[args.embedding_model_name]

    print(f'embedding model: {args.embedding_model_name}', 'threshold:', embedding_model_threshold)
    calc_auc_and_acc(exp_dir,
                        dataset_name=args.dataset_name,
                        heuristic_threshold=heuristic_thresholds[args.dataset_name],
                        embedding_model=embedding_model,
                        threshold=embedding_model_threshold
                        )
        
