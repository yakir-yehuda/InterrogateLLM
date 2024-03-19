import argparse
from multiprocessing import set_start_method

from cos_baselines import embedding_model_thresholds
from models.interrogatellm import InterrogateLLM
from language_models import GPT3, GPTEmbedding, SBert, LLamaV2


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


if __name__ == '__main__':
    set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    # arguments for the experiments (dataset, answer model, reconstruction model, embedding model, experiment number)
    parser.add_argument('--ans_model', type=str, default='llamaV2-7', choices=['gpt', 'llamaV2-7', 'llamaV2-13'], help='llm model name')
    parser.add_argument('--embedding_model_name', type=str, default='ada002', choices=['ada002', 'sbert'], help='embedding model name')
    parser.add_argument('--reconstruction_models', type=str, default='gpt,llamaV2-7,llamaV2-13', help='reconstruction models names')
    parser.add_argument('--iterations', type=int, default=2, help='iterations number')
    args = parser.parse_args()
    
    # create ans model instance
    answer_model = get_llm_model(args.ans_model)
    # create embedding model instance
    embedding_model = get_embedding_model(args.embedding_model_name)
    # create reconstruction models instances (make sure to not duplicate the answer model)
    reconstruction_models_list = [get_llm_model(model_name) if model_name != args.ans_model else answer_model for model_name in args.reconstruction_models.split(',')]
    
    # create model pipe instance
    model = InterrogateLLM(answer_model, reconstruction_models_list, embedding_model, embedding_model_thresholds[args.embedding_model_name])
    
    # few shot examples
    few_shot_examples = [("What is the capital of France?", "The capital is Paris."),
                         ("What is the capital of Japan?", "The capital is Tokyo."),
                         ("What is the capital of Australia?", "The capital is Canberra.")]
    
    
    query = "What is the capital of Germany?"
    # run model
    print(f'Running InterrogateLLM model...\n')
    res = model.model_run(query, few_shot_examples, iterations=args.iterations)
    print(f'IntterogateLLM result (True if Hallucination else False):\n{res}')