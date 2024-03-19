import utils
import pandas as pd
from datetime import datetime
from models.model import ModelPipe



class MoviesModel(ModelPipe):
    def few_shot_examples(self):
        """
        examples of question answer for movies dataset for few shot learning
        @return: return a list of question-answer examples
        """
        question_1 = "What actors played in the 1995 movie Jumanji?"
        answer_1_old = "The main cast of the 1995 movie Jumanji included Robin Williams, Jonathan Hyde, Kirsten Dunst, Bradley Pierce, " \
                       "Bonnie Hunt, Bebe Neuwirth, David Alan Grier, Patricia Clarkson, Adam Hann-Byrd, Laura Bell Bundy, James Handy, " \
                       "Gillian Barber, Brandon Obray, Cyrus Thiedeke, Gary Joseph Thorup, Leonard Zola, Lloyd Berry, Malcolm Stewart, " \
                       "Annabel Kershaw, Darryl Henriques, Robyn Driscoll, Peter Bryant, Sarah Gilson, Florica Vlad, June Lion, Brenda Lockmuller."

        answer_1 = "The main cast included Robin Williams, Jonathan Hyde, Kirsten Dunst, Bradley Pierce, " \
                   "Bonnie Hunt, Bebe Neuwirth, David Alan Grier, Patricia Clarkson, Adam Hann-Byrd, Laura Bell Bundy, James Handy, " \
                   "Gillian Barber, Brandon Obray, Cyrus Thiedeke, Gary Joseph Thorup, Leonard Zola, Lloyd Berry, Malcolm Stewart, " \
                   "Annabel Kershaw, Darryl Henriques, Robyn Driscoll, Peter Bryant, Sarah Gilson, Florica Vlad, June Lion, Brenda Lockmuller."

        question_2 = "What actors played in the 2011 movie Kingdom Come?"
        answer_2 = "Selma Blair, Edward Burns, Bruce Campbell, Lizzy Caplan, Seymour Cassel, Don Cheadle, Joanne Cook, Rachael Leigh Cook, Tom Cook, Alan Cumming, Tom DiCillo, Drake Doremus."

        question_3 = "What actors played in the 2009 movie Inglourious Basterds?"
        answer_3_old = "The actors played in the 2009 movie Inglourious Basterds are Brad Pitt, Diane Kruger, Eli Roth, Mélanie Laurent, " \
                       "Christoph Waltz, Michael Fassbender, Daniel Brühl, Til Schweiger, Gedeon Burkhard, Jacky Ido, B.J. Novak, Omar Doom."

        answer_3 = "The actors are Brad Pitt, Diane Kruger, Eli Roth, Mélanie Laurent, " \
                   "Christoph Waltz, Michael Fassbender, Daniel Brühl, Til Schweiger, Gedeon Burkhard, Jacky Ido, B.J. Novak, Omar Doom."

        qa_number = 3

        qa_list = []
        for i in range(1, qa_number + 1):
            qa_list.append((eval('question_' + str(i)), (eval('answer_' + str(i)))))

        return qa_list

    def read_dataset(self):
        movies_data_path = './datasets/The-Movies-Dataset/title_with_cast.csv'
        movies_data = pd.read_csv(movies_data_path).dropna()

        return movies_data

    def dataset_generator(self):
        """
        @return: return a generator of dataset
        """
        movies_dataset = self.read_dataset()
        for index, movie in movies_dataset.iterrows():
            if index > 4000:
                break
            # get movie data
            cast = eval(movie['cast'])
            title = movie['title']
            release_date = datetime.strptime(movie['release_date'], '%Y-%m-%d').date()

            cast_list = [actor['name'] for actor in cast]

            # question about the movie cast
            question = f'What actors played in the {release_date.year} movie {title}?'

            question_args = {'movie_year': release_date.year, 'movie_title': title}
            answer_args = {'movie_cast': cast_list}

            yield question, question_args, answer_args

    def answer_heuristic(self, predicted_answer, **kwargs):
        """
        @param predicted_answer: predicted answer from the model
        @param kwargs: additional arguments
        @return: return true if the predicted answer is correct according to the heuristics
        """

        predicted_cast = utils.spacy_extract_entities(predicted_answer)
        intersection, union = utils.calculate_intersection_and_union(kwargs['movie_cast'], predicted_cast)

        threshold = 0.8
        answer_simple_heuristic = len(intersection) / len(predicted_cast) > threshold if len(predicted_cast) != 0 else True
        return answer_simple_heuristic

    def question_heuristic(self, predicted_question, **kwargs):
        """
        @param predicted_question: predicted question from the model
        @param kwargs: additional arguments
        @return: return true if the predicted question is correct according to the heuristics
        """
        original_name = kwargs['movie_title'].lower()
        predicted_question = predicted_question.lower()
        return original_name in predicted_question
