import pandas as pd
from unidecode import unidecode
from models.model import ModelPipe


class WorldModel(ModelPipe):
    def few_shot_examples(self):
        """
        examples of question answer for books dataset for few shot learning
        @return: return a list of question-answer examples
        """
        question_1 = "What is the capital of France?"
        answer_1 = "The capital is Paris."

        question_2 = "What is the capital of Japan?"
        answer_2 = "The capital is Tokyo."


        question_3 = "What is the capital of Australia?"
        answer_3 = "The capital is Canberra."


        qa_number = 3

        qa_list = []
        for i in range(1, qa_number + 1):
            qa_list.append((eval('question_' + str(i)), (eval('answer_' + str(i)))))

        return qa_list

    def read_dataset(self):
        world_data_path = './datasets/world-data/world-data-2023.csv'

        columns = ['Country','Capital/Major City', 'Largest city']
        columns = ['Country','Capital/Major City']
        world_data = pd.read_csv(world_data_path)
        world_data = world_data[columns].dropna()

        world_data = world_data[~world_data['Country'].str.contains('�', regex=True)]
        world_data = world_data[~world_data['Capital/Major City'].str.contains('�', regex=True)]

        world_data['Country'] = world_data['Country'].apply(lambda x: unidecode(x))
        world_data['Capital/Major City'] = world_data['Capital/Major City'].apply(lambda x: unidecode(x))

        return world_data

    def dataset_generator(self):
        """
        @return: return a generator of dataset
        """
        world_dataset = self.read_dataset()
        for _, country in world_dataset.iterrows():
            # get book data
            country_name = country['Country']
            country_capital = country['Capital/Major City']

            question = f'What is the capital of {country_name}?'

            question_args = {'country_name': country_name}

            country_capital = [x.strip() for x in country_capital.split(',')]
            answer_args = {'country_capital': country_capital}

            yield question, question_args, answer_args

    def answer_heuristic(self, predicted_answer, **kwargs):
        """
        @param predicted_answer: predicted answer from the model
        @param kwargs: additional arguments
        @return: return true if the predicted answer is correct according to the heuristics
        """
        country_capital = kwargs['country_capital']
        for capital in country_capital:
            if capital.lower() in predicted_answer.lower():
                return 1
        return 0

    def question_heuristic(self, predicted_question, **kwargs):
        """
        @param predicted_question: predicted question from the model
        @param kwargs: additional arguments
        @return: return true if the predicted question is correct according to the heuristics
        """
        original_name = kwargs['country_name'].lower()
        predicted_question = predicted_question.lower()
        return original_name in predicted_question

