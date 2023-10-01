import os
from typing import Optional, List

import numpy as np
import redis
import requests
from redis.commands.search.query import Query
from requests import JSONDecodeError

EMBEDDINGS_ENDPOINT = os.getenv("EMBEDDINGS_ENDPOINT", default="https://api.eventflow.ru/v1/embeddings/")
REDIS_HOST = os.getenv("REDIS_HOST", default="127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", default="6379")
REDIS_USERNAME = os.getenv("REDIS_USERNAME")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")


def encode(lines: List[str]) -> Optional[List[List[float]]]:
    response = requests.post(EMBEDDINGS_ENDPOINT, json={"input": lines})
    try:
        json = response.json()
        response.close()
        return [entry["embedding"] for entry in json["data"]]
    except JSONDecodeError:
        response.close()
        return None


def main():
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        decode_responses=True
    )

    index_name = "idx:literature/11/zamyatin.txt"

    # https://redis.io/docs/interact/search-and-query/search/vectors/
    query = (
        Query('(*)=>[KNN 5 @vector $query_vector AS vector_score]')
        .sort_by('vector_score')
        .return_fields('vector_score', 'text')
        .dialect(2)
    )

    embeddings = encode(["Месторождение газа Гронинген принесло голландской казне сотни миллиардов евро."])

    encoded_query = embeddings[0]
    query_vector = np.array(encoded_query, dtype=np.float32).tobytes()

    # Because cosine "distance" is used as the metric, the items with the smallest "distance" are closer
    # and therefore more similar to the query.
    docs = redis_client.ft(index_name).search(query, {'query_vector': query_vector}).docs
    for doc in docs:
        print(doc)


if __name__ == '__main__':
    main()
