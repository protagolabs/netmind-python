from openai.resources.embeddings import Embeddings as OpenEmbeddings, AsyncEmbeddings as AsyncOpenEmbeddings


class Embeddings(OpenEmbeddings):
    pass


class AsyncEmbeddings(AsyncOpenEmbeddings):
    pass
