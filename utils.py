import spacy
import numpy as np
from unidecode import unidecode
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load('en_core_web_sm')

def plot_histogram(list_to_plot, save_path, file_name, color='blue', label='histogram'):
    """
    plot a histogram of a given list, and save it to a given path
    @param list_to_plot: the list to plot (can be a list of lists)
    @param save_path: the path to save the histogram
    @param file_name: the name of the file to save
    @param color: the color of the histogram (can be a list of colors)
    @param label: the label of the histogram (can be a list of labels)
    """
    bins = list(range(0, 101))
    plt.hist(list_to_plot, bins, color=color, label=label)
    plt.title(f'{file_name}')
    plt.xlabel('intersection ratio')
    plt.ylabel('Counts')
    plt.xticks(range(0, 101, 10))
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig(f'{save_path}/{file_name}.png', dpi=300, bbox_inches='tight')
    plt.cla()


def calculate_intersection_and_union(ground_truth, prediction):
    """
    calculate the intersection and union of two sets of cast names (case insensitive)
    @param ground_truth: the true cast names from the imdb dataset
    @param prediction: the predicted cast names from the GPT model
    @return: the intersection and union of the two sets
    """
    ground_truth = set(map(lambda x: x.lower(), ground_truth))
    prediction = set(map(lambda x: x.lower(), prediction))

    intersection = ground_truth.intersection(prediction)
    union = ground_truth.union(prediction)

    return intersection, union


def gpt_embedding(gpt_model, text):
    """
    get the embedding of a text from GPT model
    @param gpt_model: gpt model - using ada002 model
    @param text: text to get embedding from
    @return: embedding of the text
    """
    response = gpt_model.embedding_request(text)
    response = response['data'][0]['embedding']

    return response


def gpt_query(gpt_model, query):
    """
    function for querying the GPT model with a given query
    @param gpt_model: the GPT model
    @param query: query to the GPT model
    @return: the response of the GPT model
    """
    response = gpt_model.submit_request(query)
    response = response['choices'][0]['text'].split("\n")

    return response


def similarity_sbert(text1, text2, embedding_model):

    # Compute embedding for both lists
    embeddings1 = embedding_model.encode([text1], convert_to_tensor=True)
    embeddings2 = embedding_model.encode([text2], convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return cosine_scores.item()


def cosine_similarity_between_texts(text1, text2, embedding_model):
    """
    calculate the cosine similarity between two texts
    @param text1: first text
    @param text2: second text
    @return: cosine similarity between the two texts
    """
    # get embeddings
    embedding1 = embedding_model.submit_request(text1)
    embedding2 = embedding_model.submit_request(text2)

    embedding1_np = np.array(embedding1)
    embedding2_np = np.array(embedding2)

    # calculate cosine similarity
    cosine_similarity = np.dot(embedding1_np, embedding2_np) / (norm(embedding1_np) * norm(embedding2_np))

    return cosine_similarity


def spacy_extract_entities(text, label='PERSON'):
    """
    extract entities from text using spacy
    @param text: the text to extract entities from
    @param label: the label of the entities to extract
    @return: list of entities in the text
    """
    # text.replace('-', ' ')
    doc = nlp(text)

    # extract entities
    entities = {e.text for e in doc.ents if e.label_ == label}
    return entities


def check_entity_in_sentence(entity, sentence):
    """
    check if an entity is in a sentence
    @param entity: the entity to check
    @param sentence: the sentence to check
    @return: True if the entity is in the sentence, False otherwise
    """

    # remove accents from entity and sentence to make the check case insensitive and accent insensitive
    entity = unidecode(entity)
    sentence = unidecode(sentence)
    
    # check if entity is in sentence (case insensitive) 
    return entity.lower() in sentence.lower()
