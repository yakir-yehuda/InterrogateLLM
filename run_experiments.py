import os
import argparse
import datetime

from models.book_model import BookModel
from models.movies_model import MoviesModel
from models.world_model import WorldModel

from language_models import GPT3, GPTEmbedding, SBert, LLamaV2, Falcon


def run_exp(dataset_name, answer_model_name='gpt', embedding_model_name='ada002', iterations=5):
    """ run intterogation experiment with the given parameters

    Args:
        dataset_name: (str) name of the dataset (books, movies, world)
        answer_model_name (str): name of the answer model (gpt, llamaV2-7, llamaV2-13) - other models are can be added in the future
        reconstruction_models (list): list of the reconstruction models (gpt, llamaV2-7, llamaV2-13) - any permutation of the models
        embedding_model_name (str): name of the embedding model (ada002, sbert) for the question models similarity
        iterations (list): list of the number of iterations for each reconstruction model 
    """

    
    # get model according to the model name (gpt, llamaV2-7, llamaV2-13, falcon) - other models can be added in the future
    def get_llm_model(model_name: str):
        if model_name == 'gpt':
            return GPT3()
        elif model_name == 'llamaV2-7':
            return LLamaV2(model_size=7)
        elif model_name == 'llamaV2-13':
            return LLamaV2(model_size=13)
        elif model_name == 'llamaV2-70':
            return LLamaV2(model_size=70)
        elif model_name == 'falcon':
            return Falcon()
        else:
            raise ValueError(f'unknown llm model name: {model_name}')

    # get embedding model according to the model name (ada002, sbert) - other models can be added in the future
    def get_embedding_model(model_name: str):
        if model_name == 'ada002':
            return GPTEmbedding()
        elif model_name == 'sbert':
            return SBert()
        else:
            raise ValueError(f'unknown embedding model name: {model_name}')

    # get the intterogate model according to the dataset name (books, movies, world)
    def get_intterogate_model(dataset_name: str):
        if dataset_name == 'books':
            return BookModel
        elif dataset_name == 'movies':
            return MoviesModel
        elif dataset_name == 'world':
            return WorldModel
        else:
            raise ValueError(f'unknown dataset name: {dataset_name}')

    reconstruction_models =  ['gpt', 'llamaV2-7', 'llamaV2-13']

    # create ans model instance
    ans_model = get_llm_model(answer_model_name)
    # create question models instances (list of models or single model)
    reconstruction_models_list = [get_llm_model(model_name) if model_name != answer_model_name else ans_model for model_name in reconstruction_models]
    # create embedding model instance
    embedding_model = get_embedding_model(embedding_model_name)
    # create intterogate model instance (receive answer model, reconstruction models, embedding model and iterations for each model)
    model = get_intterogate_model(dataset_name)(answer_model=ans_model,
                                    reconstruction_models=reconstruction_models_list,
                                    embedding_model=embedding_model,
                                    iterations=iterations)
    
    # create the save path for the experiment
    exp_dir = f'{dataset_name}_experiments'
    save_dir = os.path.join('.', exp_dir, embedding_model_name, answer_model_name, '-'.join(reconstruction_models))
    
    # create the save directory if not exist
    os.makedirs(save_dir, exist_ok=True)

    # save the experiment details in a file (experiment_details.txt) in the save directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(os.path.join(save_dir, 'experiment_details.txt'), 'w') as f:
        print(f'Experiment - {timestamp}\n', file=f)
        print(f'Answer model:\n     {answer_model_name}', file=f)
        print(f'Question models:', file=f)
        for model_name, iteration in zip(reconstruction_models, [iterations]*len(reconstruction_models)):
            print(f'    {model_name}, K={iteration}', file=f)
        print(f'Embedding model:\n      {embedding_model_name}', file=f)

    # run the experiment
    results = model.model_run(save_path=save_dir)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments for the experiments (dataset, answer model, reconstruction model, embedding model, experiment number)
    parser.add_argument('--dataset_name', type=str, default='world', choices=['books', 'movies', 'world'], help='dataset name')
    parser.add_argument('--ans_model', type=str, default='gpt', choices=['gpt', 'llamaV2-7', 'llamaV2-13'], help='llm model name')
    parser.add_argument('--embedding_model_name', type=str, default='ada002', choices=['ada002', 'sbert'], help='embedding model name')
    args = parser.parse_args()

    # run experiment
    res = run_exp(dataset_name=args.dataset_name, 
                  answer_model_name=args.ans_model, 
                  embedding_model_name=args.embedding_model_name)

