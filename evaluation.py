import rouge
import pandas as pd

def rouge_eval(hypothesis,references):
    """
    ROUGE
    source: https://github.com/Diego999/py-rouge#readme

    Keyword arguments:
    hypothesis: automatically produced summary
    references: human written summary

    Return:
    a dataframe with different metrics
    """
    precision = []
    recall = []
    fscore = []
    metrics = []
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                        max_n=1,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=False,
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)

    scores = evaluator.get_scores(hypothesis, references)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        for method_id, results_per_ref in enumerate(results):
            nb_references = len(results_per_ref['p'])
            for reference_id in range(nb_references):
                precision.append("{:.2f}".format(100*results_per_ref['p'][reference_id]))
                recall.append("{:.2f}".format(100*results_per_ref['r'][reference_id]))
                fscore.append("{:.2f}".format(100*results_per_ref['f'][reference_id]))
    return({"P":precision,"R":recall,"F":fscore})