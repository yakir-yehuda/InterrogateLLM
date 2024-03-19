import re
import os
import abc
import pickle as pkl

from tqdm import tqdm
from multiprocessing import Pool
from sentence_transformers import util



def question_pred(model, prompt, model_name, temperatures):
    res = []
    for temp in temperatures:
        response = model.submit_request(prompt, temperature=temp, split_by='Answer:')
        response = [res for res in response if res != '']
        response = ', '.join(response)
        res.append((temp, response))
        
    return {model_name:res}

# reconstruct question from the answer using the reconstruction models (gpt, llamaV2-7, llamaV2-13) using multiprocessing pool for parallelism 
def reconstruct_pool(models, prompt, temperatures):
    with Pool() as pool:
        tmp = pool.starmap(question_pred, [(models[0], prompt, 'gpt', temperatures), (models[1], prompt, 'l7', temperatures), (models[2], prompt, 'l13', temperatures)])
    return tmp


class ModelPipe:
    def __init__(self, answer_model, reconstruction_models, embedding_model, iterations, t_0=0.7):
        self.answer_model = answer_model
        self.reconstruction_models = reconstruction_models
        self.embedding_model = embedding_model
        self.iterations = iterations
        self.t_0 = t_0

    @abc.abstractmethod
    def few_shot_examples(self):
        """
        @return: return a list of few shot question answer examples according to the dataset
        """
        pass

    @abc.abstractmethod
    def read_dataset(self,):
        """
        read dataset from file
        @return: return dataset
        """
        pass

    @abc.abstractmethod
    def dataset_generator(self):
        """
        @return: return a generator of dataset
        """
        pass

    @abc.abstractmethod
    def answer_heuristic(self, predicted_answer, **kwargs):
        """
        @param predicted_answer: predicted answer from the model
        @param kwargs: additional arguments
        @return: return true if the predicted answer is correct according to the heuristics
        """
        pass

    @abc.abstractmethod
    def question_heuristic(self, predicted_question, **kwargs):
        """
        @param predicted_question: predicted question from the model
        @param kwargs: additional arguments
        @return: return true if the predicted question is correct according to the heuristics
        """
        pass

    @staticmethod
    def create_few_shot_prompt(question_answer_list, inverse=False):
        """
        create a few shot prompt according to input QA list
        @param question_answer_list: question answer example list for few shot
        @param inverse: if true, create inverse qa few shot (answer question instead question answer)
        @return: return a few shot prompt with len(qa_list) shots
        """
        few_shot_prompt = ""
        for question, answer in question_answer_list:
            question = 'Question: ' + question
            answer = 'Answer: ' + answer
            if inverse:
                few_shot_prompt += answer + "\n" + question + "\n\n"
            else:
                few_shot_prompt += question + "\n" + answer + "\n\n"

        return few_shot_prompt


    def reconstruct_question(self, predicted_answer, inverse_prompt_prefix , temperatures):
        answer_question_instructions = 'Follow the format below, and please only predict the question that corresponds to the last answer.\n\n'
        answer_question_prompt = answer_question_instructions + inverse_prompt_prefix + 'Answer: ' + predicted_answer + '\n' + 'Question: '
        
        res = reconstruct_pool(self.reconstruction_models, answer_question_prompt, temperatures)
        return res

    def model_run(self, save_path='.'):
        # create few shot prompt according to the few shot examples of the dataset
        few_shot_examples = self.few_shot_examples()
        # create instructions for the model
        question_answer_instructions = 'Follow the format below, and please only predict the answer that corresponds to the last question.\n\n'
        # prompt prefix (instructions + few shot examples)
        prompt_prefix = question_answer_instructions + self.create_few_shot_prompt(few_shot_examples)
        # create inverse prompt prefix (instructions + few shot examples) for predicting questions from the answer (reconstruction step)
        inverse_prompt_prefix = self.create_few_shot_prompt(few_shot_examples, inverse=True)

        results = []
        index = 0
        skip_index = 0
        
        # check if there are already results in the save path (if the experiment was stopped and we want to continue from the last point)
        if os.path.isdir(os.path.join(save_path, 'res_pkl')):
            current_dir = os.path.join(save_path, 'res_pkl')
            pickle_files_num = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(current_dir)]
            # get the last index of the results
            skip_index = max(pickle_files_num)    

            file_path = os.path.join(current_dir, 'res_' + str(skip_index) + '.pkl')
            with open(file_path, 'rb') as handle:
                results = pkl.load(handle)

        # iterate over the dataset (if we want to continue from the last point, we skip the first skip_index examples)
        for question, questions_args, answer_args in tqdm(self.dataset_generator()):
            if index < skip_index:
                index += 1
                continue
            
            index += 1
            # add few shot prefix to the question to create the prompt
            prompt = prompt_prefix + 'Question: ' + question + '\n' + 'Answer: '
            # submit the prompt to the model
            response = self.answer_model.submit_request(prompt)
            # remove empty strings from response
            response = [res for res in response if res != '']
            # remove leading '-' from the response
            response = list(map(lambda x: re.sub("^([-]*)", "", x), response))
            # concatenate the response to a single string
            predicted_answer = ', '.join(response)

            if len(response) == 0:
                print('Response is empty')
                continue

            # predict questions from the answer (get list of questions according to the answer)
            predicted_questions_const = self.reconstruct_question(predicted_answer, inverse_prompt_prefix, [self.t_0] * self.iterations)
            predicted_questions_var = self.reconstruct_question(predicted_answer, inverse_prompt_prefix, [self.t_0 + (1-self.t_0) *i/self.iterations for i in range(0,self.iterations)])
            
            # get embedding vector for original question
            original_question_embedding = self.embedding_model.submit_embedding_request(question)
            
            # get embedding vector for each reconstructed question and calculate cosine similarity (const temperature)
            pred_questions_embedding_const = []
            for model_pred in predicted_questions_const:
                model_name = list(model_pred.keys())[0]
                model_pred_res = model_pred[model_name]
                res = []
                for temp, pred_question in model_pred_res:
                    embedding_pred_question = self.embedding_model.submit_embedding_request(pred_question)
                    questions_cosine_similarity = util.cos_sim(original_question_embedding, embedding_pred_question).item()
                    res.append((temp, pred_question, questions_cosine_similarity))
                pred_questions_embedding_const.append({model_name:res})
            
            # get embedding vector for each reconstructed question and calculate cosine similarity (variable temperature)
            pred_questions_embedding_var = []
            for model_pred in predicted_questions_var:
                model_name = list(model_pred.keys())[0]
                model_pred_res = model_pred[model_name]
                res = []
                for temp, pred_question in model_pred_res:
                    embedding_pred_question = self.embedding_model.submit_embedding_request(pred_question)
                    questions_cosine_similarity = util.cos_sim(original_question_embedding, embedding_pred_question).item()
                    res.append((temp, pred_question, questions_cosine_similarity))
                pred_questions_embedding_var.append({model_name:res})

            log = {'answer_args': answer_args,
                   'predicted_answer': predicted_answer,
                   'original_question': question,
                   'question_args': questions_args,
                   'predicted_questions_const': pred_questions_embedding_const,
                   'predicted_questions_var': pred_questions_embedding_var,
                   }

            results.append(log)

            # save the results every x examples
            if len(results) % 100 == 0:
                dir_path = os.path.join(save_path, f'res_pkl')
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, f'res_{len(results)}.pkl')
                with open(file_path, 'wb') as handle:
                    pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

        return results
