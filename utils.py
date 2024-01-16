import base64
import heapq
import logging
import os
import re
import time

import openai
import tiktoken
from openai.error import ServiceUnavailableError, InvalidRequestError, APIError, APIConnectionError, Timeout, RateLimitError


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE", default="https://api.openai.com/v1")


def num_tokens_from_messages(prompt, model="gpt-3.5-turbo-16k-0613"):
    """Return the number of tokens used by a list of messages."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value, disallowed_special=()))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def gpt_call(prompt: str, model: str = "gpt-3.5-turbo", max_retries: int = 100, temperature: int = 0, seed: int = 0):
    messages = [
        {"role": "user", "content": prompt}
    ]
    for i in range(max_retries):
        try:
            result = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed
            )
            return result
        except InvalidRequestError as e:
            logging.warning(e)
            return e.user_message
        except (ServiceUnavailableError, APIError, APIConnectionError, Timeout, RateLimitError) as e:
            logging.warning(f"Retry {i + 1}/{max_retries}: {e}")
            time.sleep(1)
    logging.error("Exceeded maximum retry number")


def text_to_split(text):
    text = [x for x in re.split(r'(\W)', text.lower()) if x.strip()]
    return text


def get_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def jaccard_similarity(sample, corpus_sets, top_n):
    sample = set(text_to_split(sample))

    similarities = []
    for corpus_idx, corpus_set in enumerate(corpus_sets):
        similarity = get_jaccard_similarity(sample, corpus_set)
        heapq.heappush(similarities, (similarity, corpus_idx))
        if len(similarities) > top_n:
            heapq.heappop(similarities)

    top_similarities = [(idx, sim) for sim, idx in similarities]
    top_similarities.sort(key=lambda x: x[1], reverse=False)
    return top_similarities
