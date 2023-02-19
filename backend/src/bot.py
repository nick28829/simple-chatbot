"""Simple chatbot using a yaml file to configure the possible intents.

It is built to be useable without a huge dataset. Therefore, it uses
example example sentences to compare a given statement to. They are specified
in the config file. The comparison uses a word2vec model and cosine similiarity.

If you execute this file directly, a cli prompt is started where you can test
the bot without a need to run the webinterface. Beware that loading (and the
first time downloading) the w2v model as well as the stopwords takes some time.

If you are reading the code, I'm sorry, there is still some stuff that I'll need
to clean and improve. This isn't really a production ready software but just
something I built when it was rainy and I wanted to see if I could.
"""

from typing import Any, Callable, Iterable, List, Tuple
from collections import Counter

import gensim
import gensim.downloader
import nltk
import numpy as np
from yaml import load, Loader
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


def load_w2v() -> gensim.models.Word2Vec:
    """Load (download if necessary) the Word2Vec model.

    Returns:
        gensim.models.Word2Vec: w2v model.
    """
    return gensim.downloader.load('word2vec-google-news-300')


def tokenize(sentence: str) -> List[str]:
    """Tokenize sentence and remove stop words.

    Args:
        sentence (str): Sentence to tokenize.

    Returns:
        List[str]: List with tokens.
    """
    normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
    tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
    return [t for t in tokens if t not in STOP_WORDS]


def vectorize_phrase(sentence: str, model: gensim.models.Word2Vec=None) -> np.array:
    """Vectorize a phrase by first tokenizing it, apply the vectorization
    using `model` and then averaging the vector over all tokens excluding stopwords.

    Args:
        sentence (str): Sentence to vectorize.
        model (_type_, optional): Word embedding Model. Defaults to None.

    Returns:
        np.array: Vector embedding for the sentence.
    """
    tokens = tokenize(sentence)
    if not tokens:
        return 0
    token_count = Counter(tokens)
    weights = [token_count[token] for token in token_count if token in model]
    if not weights:
        return None
    return np.average([model[token] for token in token_count if token in model], axis=0, weights=weights).reshape(1, -1)


def first_match(l: Iterable[Any], f: Callable) -> Any:
    """Iterate over `l`, check againt `f` and return the first match.

    Args:
        l (Iterable[Any]): Iterable to search in.
        f (Callable): Function to return a `bool` for each element in `l`.

    Returns:
        Any: Either the first match or `None`.
    """
    for list_element in l:
        if f(list_element):
            return list_element
    return None


class BadRequest(Exception):
    """Exception raised if there is an error in the input.
    """
    pass


class Intent:
    """Intent of the chatbot.
    """

    def __init__(self, model: gensim.models.Word2Vec=None, **kwargs):
        """Set intent details provided in config file and given as `kwargs`.

        Args:
            model (gensim.models.Word2Vec, optional): Model to use for vectorization
            of words. Defaults to None.
        """
        self.model = model
        self.name = kwargs['name'] if 'name' in kwargs else None
        self.response = kwargs['response'] if 'response' in kwargs else None
        self.example_phrases = kwargs['example_phrases'] if 'example_phrases' in kwargs else []
        self.parameters = kwargs['parameters'] if 'parameters' in kwargs else [] # List[dict]
        self.data = kwargs['data'] if 'data' in kwargs else None
        self.action = kwargs['action'] if 'action' in kwargs else None

        for phrase in self.example_phrases:
            self.phrase_vecs = vectorize_phrase(phrase, self.model) # FIXME: Bug! Only storing one example sentence

    def compare(self, sentence_vec: np.array) -> Tuple[float]:
        """Compare `sentence_vec` against example sentences and return maximum
        and average cosine similiarity.

        Args:
            sentence_vec (np.array): Vector represeantation of sentence to compare.

        Returns:
            Tuple[float]: Maximum and average cosine similiarity.
        """
        if self.phrase_vecs is not None:
            comp_scores = [cosine_similarity(sentence_vec.reshape(1, -1), ref_vec.reshape(1, -1))[0][0] for ref_vec in self.phrase_vecs]
            return max(comp_scores), sum(comp_scores) / len(comp_scores)
        return 0, 0

    @classmethod
    def get_parameter_from_string(cls, sentence: str, parameter: dict) -> str:
        """Extract parameter value from `sentence` using string similiarity.

        Args:
            sentence (str): Sentence containing parameter.
            parameter (dict): Key of parameter to extract.

        Returns:
            str: Extracted parameter value.
        """
        sentence = sentence.lower()
        for value in parameter['values']:
            if fuzz.partial_ratio(sentence, value.lower()) > 80:
                return value
        return None

    def build_answer(self, context: 'Context') -> str:
        """Build an answer based on the given context.

        Args:
            context (Context): Context including last given statement with extracted parameters.

        Returns:
            str: Answer to return to user.
        """
        answer = self.response
        if self.data:
            # get data with matchin parameters in context
            data = first_match(self.data, lambda el: all(el['parameters'][p['name']] == p['value'] for p in context.intent_args))
            if data:
                for key, value in data['data'].items():
                    answer = answer.replace(f'${key}', value)
        return answer

    def process(self, sentence: str, context: 'Context') -> Tuple[str, 'Context']:
        """Process a new user input based on the provided `sentence` and the previous `context`.

        Args:
            sentence (str): Statement provided by user.
            context (Context): Context provided with user request.

        Returns:
            Tuple[str, 'Context']: Answer to the user and the updated context.
        """
        # get all the needed entities from sentence
        # if missing entities, ask for them
        if context.intent and context.missing_args:
            for parameter in context.missing_args:
                value = self.get_parameter_from_string(sentence, first_match(self.args, lambda el: el['name'] == context.intent))
                if value:
                    context.intent_args.append({
                        'value': value,
                        'name': parameter
                    })
                    context.missing_args.remove(parameter)
        
        if not context.intent and self.parameters:
            # try getting missing entities from sentence
            for parameter in self.parameters:
                if not context.intent_args or not first_match(context.intent_args, lambda el: el['value'] == parameter['name']):
                    value = self.get_parameter_from_string(sentence, parameter)
                    if value:
                        context.intent_args.append({
                            'value': value,
                            'name': parameter['name']
                        })
                    else:
                        context.missing_args.append(parameter['name'])
        if context.missing_args:
            return first_match(self.parameters, lambda el: el['name'] in context.missing_args)['follow_up'], context.to_dict()
        if self.action:
            pass # TODO: implement action taken
        answer = self.build_answer(context)
        return answer, context.to_dict()


class Context:
    """Context of a user request. It contains information about past requests,
    missing arguments, etc.
    """

    def __init__(self, context: dict):
        self.follow_up = context['follow_up'] if 'follow_up' in context else False # type: bool
        self.intent_args = context['intent_args'] if 'intent_args' in context else [] # type: list
        self.missing_args = context['missing_args'] if 'missing_args' in context else []  # type: list
        self.intent_history = context['intent_history'] if 'intent_history' in context else [] # type: list
        self.intent = context['intent'] if 'intent' in context else None # type: str

    def __bool__(self) -> bool:
        return self.follow_up or bool(self.intent_args)

    def to_dict(self) -> dict:
        """Create a dictionary representation that can be converted to JSON to be sent to the user.

        Returns:
            dict: Dictionary representation.
        """
        return {
            'follow_up': self.follow_up,
            'intent_args': self.intent_args,
            'missing_args': self.missing_args,
            'intent_history': self.intent_history,
        }


class Bot:
    """Bot class managing the different intents and high level actions.
    """

    def __init__(self, config_file: str, model: gensim.models.Word2Vec=None):
        """Initialize Bot.

        Args:
            config_file (str): path to the config file in `yaml` format.
            model (gensim.models.Word2Vec, optional): Model to use for word vectorization.
            Defaults to None.
        """
        self.model = model
        self.intents = []  # type: List[Intent]
        with open(config_file, 'r') as f:
            config = load(f, Loader=Loader)
            for intent_conf in config['intents']:
                self.intents.append(Intent(self.model, **intent_conf))
            self.fallback_response = config['general_responses']['fallback_response'] if 'fallback_response' in config['general_responses'] else ''
            self.conversation_stimulus = config['general_responses']['conversation_stimulus'] if 'conversation_stimulus' in config['general_responses'] else ''
            self.greeting = config['general_responses']['greeting'] if 'greeting' in config['general_responses'] else ''
        self.replacement_entities = {}
        for intent in self.intents:
            for parameter in intent.parameters:
                for value in parameter['values']:
                    self.replacement_entities[value] = parameter['name']


    def find_intent(self, sentence_vec: np.array) -> Intent:
        """Find the best matching intent for a given `sentence_vec`.

        Args:
            sentence_vec (np.array): Vectorized sentence to compare intents to.

        Returns:
            Intent: Best matching intent or `None` if noe is suitable.
        """
        scores = []
        for intent in self.intents:
            max_val, avg_val = intent.compare(sentence_vec)
            scores.append((max_val, avg_val))
        max_score, index = 0, 0
        for current_index, (m, _) in enumerate(scores):
            if m > max_score:
                index = current_index
                max_score = m
        if max_score < 0.35:
            return None
        return self.intents[index]

    def replace_sentence_entities(self, sentence: str) -> str:
        """Replace entities in a sentence with the generalization term for better comparibility.

        Args:
            sentence (str): Sentence to replace entities in.

        Returns:
            str: Generalized sentence.
        """
        for entity, generalization in self.replacement_entities.items():
            if entity in sentence:
                sentence = sentence.replace(entity, generalization)
        return sentence

    def get_response(self, sentence: str, context: Context) -> Tuple[str, Context]:
        """Create a response based on `sentence` and the `context` from previous requests.

        Args:
            sentence (str): Statement given by the user.
            context (Context): context of previous requests.

        Raises:
            BadRequest: If an error in the context is detected.

        Returns:
            Tuple[str, Context]: Answer, new context.
        """
        if not sentence and not context:
            return self.greeting, context.to_dict()
        if context.follow_up:
            if context.intent:
                intent = first_match(self.intents, lambda el: el.name == context.intent)
                if not intent:
                    raise BadRequest
            else:
                raise BadRequest
        else:
            sentence_generalization = self.replace_sentence_entities(sentence)
            sentence_vec = vectorize_phrase(sentence_generalization, self.model)
            if sentence_vec is not None:
                intent = self.find_intent(sentence_vec)
            else:
                intent = None
        if not intent:
            return self.fallback_response, context.to_dict()
        return intent.process(sentence, context)
        

if __name__=='__main__':
    model = load_w2v()
    bot = Bot('data.yaml', model)
    context = {}
    user_in = ''
    while True:
        context = Context(context)
        answer, context = bot.get_response(user_in, context)
        print(answer)
        user_in = input()
