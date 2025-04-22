from datasets import load_metric
import evaluate
import json

BLEU = evaluate.load("bleu")


def evaluate_model(predictions_file):
    with open(predictions_file) as f:
        data = [json.loads(line) for line in f]

    references = [d["response"] for d in data]
    predictions = [d.get("prediction", "") for d in data]

    results = BLEU.compute(predictions=predictions, references=[[ref] for ref in references])
    print("BLEU score:", results)
    return results

if __name__ == "__main__":
    evaluate_model("evaluation/generated_predictions.jsonl")
