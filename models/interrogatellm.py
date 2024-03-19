import re
import numpy as np
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
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


class InterrogateLLM:
    def __init__(self, answer_model, reconstruction_models, embedding_model, embedding_threshold, t_0=0.7):
        self.answer_model = answer_model
        self.reconstruction_models = reconstruction_models
        self.embedding_model = embedding_model
        self.embedding_threshold = embedding_threshold
        self.t_0 = t_0


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

    def model_run(self, query, few_shot_examples, iterations=5, variable_temp=False):
        # create instructions for the model
        question_answer_instructions = 'Follow the format below, and please only predict the answer that corresponds to the last question.\n\n'
        # prompt prefix (instructions + few shot examples)
        prompt_prefix = question_answer_instructions + self.create_few_shot_prompt(few_shot_examples)
        # create inverse prompt prefix (instructions + few shot examples) for predicting questions from the answer (reconstruction step)
        inverse_prompt_prefix = self.create_few_shot_prompt(few_shot_examples, inverse=True)


        # add few shot prefix to the question to create the prompt
        prompt = prompt_prefix + 'Question: ' + query + '\n' + 'Answer: '
        # submit the prompt to the model
        response = self.answer_model.submit_request(prompt, split_by='Question:')
        # remove empty strings from response
        response = [res for res in response if res != '']
        # remove leading '-' from the response
        response = list(map(lambda x: re.sub("^([-]*)", "", x), response))
        # concatenate the response to a single string
        predicted_answer = ', '.join(response)
        
        if len(response) == 0:
            print('Response is empty')

        temperatures = [self.t_0] * iterations if not variable_temp else [self.t_0 + (1-self.t_0) * i/iterations for i in range(0, iterations)]
        # predict questions from the answer (get list of questions according to the answer)
        predicted_questions = self.reconstruct_question(predicted_answer, inverse_prompt_prefix, temperatures)

        # get embedding vector for original question
        original_question_embedding = self.embedding_model.submit_embedding_request(query)
        
        # get embedding vector for each reconstructed question and calculate cosine similarity (const temperature)
        pred_questions_embedding = []
        for model_pred in predicted_questions:
            model_name = list(model_pred.keys())[0]
            model_pred_res = model_pred[model_name]
            res = []
            for temp, pred_question in model_pred_res:
                embedding_pred_question = self.embedding_model.submit_embedding_request(pred_question)
                questions_cosine_similarity = util.cos_sim(original_question_embedding, embedding_pred_question).item()
                res.append((temp, pred_question, questions_cosine_similarity))
            pred_questions_embedding.append({model_name:res})
        
        cosine_scores = [score for res in pred_questions_embedding for key, value in res.items() for _, _, score in value]
        avg_cosine_score = np.average(cosine_scores)
        

        print(f'Query:\n{query}')
        print(f'Predicted Answer:\n{predicted_answer}\n')

        return avg_cosine_score < self.embedding_threshold
