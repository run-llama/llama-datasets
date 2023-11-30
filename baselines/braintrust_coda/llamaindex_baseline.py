import nest_asyncio
import tqdm

from llama_index.llama_dataset import download_llama_dataset
from llama_index.llama_pack import download_llama_pack
from llama_index import VectorStoreIndex

nest_asyncio.apply()

# DOWNLOAD LLAMADATASET
rag_dataset, documents = download_llama_dataset(
   "BraintrustCodaDataset", "./paul_graham"
)

# BUILD BASIC RAG PIPELINE
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# EVALUATE WITH PACK
RagEvaluatorPack = download_llama_pack(
  "RagEvaluatorPack", "./pack_stuff"
)
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine,
    rag_dataset=rag_dataset
)
benchmark_df = await rag_evaluator.run()
print(benchmark_df)