import pandas as pd
from models.model import ModelPipe


class BookModel(ModelPipe):
    def few_shot_examples(self):
        """
        examples of question answer for books dataset for few shot learning
        @return: return a list of question-answer examples
        """
        question_1 = "Who is the author of the book Classical Mythology, what year was it published?"
        answer_1 = "The author is Mark P. O. Morford, and it was published in 2002."

        question_2 = "Who is the author of the book Clara Callan, what year was it published?"
        answer_2 = "The author is Richard Bruce Wright, and it was published in 2001."

        question_3 = "Who is the author of the book Decision in Normandy, what year was it published?"
        answer_3 = "The author is Carlo D'Este, and it was published in 1991."

        qa_number = 3

        qa_list = []
        for i in range(1, qa_number + 1):
            qa_list.append((eval('question_' + str(i)), (eval('answer_' + str(i)))))

        return qa_list

    def read_dataset(self):
        books_data_path = './datasets/books_data/books_filtered.csv'
        books_data = pd.read_csv(books_data_path)

        return books_data

    def dataset_generator(self):
        """
        @return: return a generator of dataset
        """
        books_dataset = self.read_dataset()
        for index, book in books_dataset.iterrows():
            if index > 4000:
                break
            # get book data
            book_title = eval(book['"Book-Title"'])
            book_author = eval(book['"Book-Author"'])
            book_year = eval(book['"Year-Of-Publication"'])
            book_publisher = eval(book['"Publisher"'])

            question = f'Who is the author of the book {book_title}, what year was it published?'

            question_args = {'book_title': book_title}
            answer_args = {'book_author': book_author, 'book_year': book_year, 'book_publisher': book_publisher}

            yield question, question_args, answer_args

    def answer_heuristic(self, predicted_answer, **kwargs):
        """
        @param predicted_answer: predicted answer from the model
        @param kwargs: additional arguments
        @return: return true if the predicted answer is correct according to the heuristics
        """
        answer_simple_heuristic = sum([1 for x in kwargs.values() if x.lower() in predicted_answer.lower()])

        return answer_simple_heuristic

    def question_heuristic(self, predicted_question, **kwargs):
        """
        @param predicted_question: predicted question from the model
        @param kwargs: additional arguments
        @return: return true if the predicted question is correct according to the heuristics
        """
        original_name = kwargs['book_title'].lower()
        predicted_question = predicted_question.lower()
        # TODO: check
        return original_name in predicted_question or original_name.split(':')[0] in predicted_question
