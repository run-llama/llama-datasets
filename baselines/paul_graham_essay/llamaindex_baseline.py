import nest_asyncio
import tqdm

from llama_index.llama_dataset import download_llama_dataset
from llama_index import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator
)
from llama_index.evaluation.notebook_utils import (
    get_eval_results_df,
)


nest_asyncio.apply()

# DOWNLOAD LLAMADATASET
rag_dataset, documents = download_llama_dataset(
   "PaulGrahamEssayDataset", "./paul_graham"
)

# BUILD BASIC RAG PIPELINE
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()

# PREDICT LLAMADATASET WITH RAG PIPELINE
prediction_dataset = await rag_dataset.amake_predictions_with(
    query_engine=query_engine,
    show_progress=True
)

# EVALUATION
judges = {}
judges["correctness"] = CorrectnessEvaluator(
    service_context=ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4"),
    )
)

judges["relevancy"] = RelevancyEvaluator(
    service_context=ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4"),
    )
)

judges["faithfulness"] = FaithfulnessEvaluator(
    service_context=ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model="gpt-4"),
    )
)

judges["semantic_similarity"] = SemanticSimilarityEvaluator(
    service_context=ServiceContext.from_defaults()
)

evals = {
    "correctness": [],
    "relevancy": [],
    "faithfulness": [],
    "context_similarity": [],
}

for example, prediction in tqdm.tqdm(zip(
        rag_dataset.examples,
        prediction_dataset.predictions
    )
):
    correctness_result = judges["correctness"].evaluate(
        query=example.query,
        response=prediction.response,
        reference=example.reference_answer,
    )

    relevancy_result = judges["relevancy"].evaluate(
        query=example.query,
        response=prediction.response,
        contexts=prediction.contexts,
    )

    faithfulness_result = judges["faithfulness"].evaluate(
        query=example.query,
        response=prediction.response,
        contexts=prediction.contexts,
    )

    semantic_similarity_result = judges["semantic_similarity"].evaluate(
        query=example.query,
        response="\n".join(prediction.contexts),
        reference="\n".join(example.reference_contexts),
    )

    evals["correctness"].append(correctness_result)
    evals["relevancy"].append(relevancy_result)
    evals["faithfulness"].append(faithfulness_result)
    evals["context_similarity"].append(semantic_similarity_result)

deep_eval_df, mean_correctness_df = get_eval_results_df(
    ["base_rag"]*len(evals["correctness"]), evals["correctness"],
    metric="correctness"
)
_, mean_relevancy_df = get_eval_results_df(
    ["base_rag"]*len(evals["relevancy"]), evals["relevancy"],
    metric="relevancy"
)
_, mean_faithfulness_df = get_eval_results_df(
    ["base_rag"]*len(evals["faithfulness"]), evals["faithfulness"],
    metric="faithfulness"
)
_, mean_context_similarity_df = get_eval_results_df(
    ["base_rag"]*len(evals["context_similarity"]),
    evals["context_similarity"], metric="context_similarity"
)

mean_scores_df = pd.concat(
    [
        mean_correctness_df.reset_index(),
        mean_relevancy_df.reset_index(),
        mean_faithfulness_df.reset_index(),
        mean_context_similarity_df.reset_index(),
    ],
    axis=0,
    ignore_index=True,
)
mean_scores_df = mean_scores_df.set_index("index")
mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])
print(mean_scores_df)
