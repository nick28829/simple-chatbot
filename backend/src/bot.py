from typing import Callable, Iterable
from collections import Counter

import gensim.downloader
import nltk
import numpy as np
from yaml import load, Loader
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


def load_w2v():
    return gensim.downloader.load('word2vec-google-news-300')


def tokenize(sentence):
    normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
    tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
    return [t for t in tokens if t not in STOP_WORDS]


def vectorize_phrase(sentence: str, model=None):
    tokens = tokenize(sentence)
    if not tokens:
        return 0
    token_count = Counter(tokens)
    weights = [token_count[token] for token in token_count if token in model]
    if not weights:
        return None
    return np.average([model[token] for token in token_count if token in model], axis=0, weights=weights).reshape(1, -1)


def first_match(l: Iterable, f: Callable):
    for list_element in l:
        if f(list_element):
            return list_element
    return None


class BadRequest(Exception):
    pass


class Intent:

    def __init__(self, model=None, **kwargs):
        self.model = model
        self.name = kwargs['name'] if 'name' in kwargs else None
        self.response = kwargs['response'] if 'response' in kwargs else None
        self.example_phrases = kwargs['example_phrases'] if 'example_phrases' in kwargs else []
        self.parameters = kwargs['parameters'] if 'parameters' in kwargs else []
        self.data = kwargs['data'] if 'data' in kwargs else None
        self.action = kwargs['action'] if 'action' in kwargs else None

        for phrase in self.example_phrases:
            self.phrase_vecs = vectorize_phrase(phrase, self.model)

    def compare(self, sentence_vec):
        if self.phrase_vecs is not None:
            comp_scores = [cosine_similarity(sentence_vec.reshape(1, -1), ref_vec.reshape(1, -1))[0][0] for ref_vec in self.phrase_vecs]
            return max(comp_scores), sum(comp_scores) / len(comp_scores)
        return 0, 0

    @classmethod
    def get_parameter_from_string(cls, sentence: str, parameter: dict) -> str:
        sentence = sentence.lower()
        for value in parameter['values']:
            if fuzz.partial_ratio(sentence, value.lower()) > 0.8:
                return value
        return None

    def build_answer(self, context: 'Context'):
        answer = self.response
        if self.data:
            # get data with matchin parameters in context
            data = first_match(self.data, lambda el: all(el['parameters'][p['name']] == p['value'] for p in context.intent_args))
            if data:
                for key, value in data['data'].items():
                    answer = answer.replace(f'${key}', value)
        return answer

    def process(self, sentence: str, context: 'Context'):
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

    def __init__(self, context: dict):
        self.follow_up = context['follow_up'] if 'follow_up' in context else False # type: bool
        self.intent_args = context['intent_args'] if 'intent_args' in context else [] # type: list
        self.missing_args = context['missing_args'] if 'missing_args' in context else []  # type: list
        self.intent_history = context['intent_history'] if 'intent_history' in context else [] # type: list
        self.intent = context['intent'] if 'intent' in context else None # type: str

    def __bool__(self) -> bool:
        return self.follow_up or bool(self.intent_args)

    def to_dict(self) -> dict:
        return {
            'follow_up': self.follow_up,
            'intent_args': self.intent_args,
            'missing_args': self.missing_args,
            'intent_history': self.intent_history,
        }


class Bot:

    def __init__(self, config_file: str, model=None):
        self.model = model
        self.intents = []
        with open(config_file, 'r') as f:
            config = load(f, Loader=Loader)
            for intent_conf in config['intents']:
                self.intents.append(Intent(self.model, **intent_conf))
            self.fallback_response = config['general_responses']['fallback_response'] if 'fallback_response' in config['general_responses'] else ''
            self.conversation_stimulus = config['general_responses']['conversation_stimulus'] if 'conversation_stimulus' in config['general_responses'] else ''
            self.greeting = config['general_responses']['greeting'] if 'greeting' in config['general_responses'] else ''

    def find_intent(self, sentence_vec):
        scores = []
        for intent in self.intents:
            max_val, avg_val = intent.compare(sentence_vec)
            scores.append((max_val, avg_val))
        max_score, index = 0, 0
        for current_index, (m, _) in enumerate(scores):
            if m > max_score:
                index = current_index
                max_score = m
        print(max_score)
        if max_score < 0.35:
            return None
        return self.intents[index]

    def get_response(self, sentence: str, context: Context):
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
            sentence_vec = vectorize_phrase(sentence, self.model)
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
