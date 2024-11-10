import evaluate
import datasets
from argparse import ArgumentParser
from glob import glob
import os
import json
import re

def get_references(dataset_name):
    
    references =[]
    
    dataset = datasets.load_dataset(dataset_name, cache_dir="./data")
    
    if dataset_name == "xsum":
        references = dataset['test']['summary']
    
    elif 'science' in dataset_name:
        references = dataset['test']['related_work']
    
    return references
            
def post_process(dataset_name, preds, refs):
    
    new_preds = []
    new_refs = []
    if 'science' in dataset_name:
        
        cite_text = "cite_\d+"
        p = re.compile(cite_text)
        
        for pred, ref in zip(preds, refs):
            ref_cite_texts = set(p.findall(ref))
            for cite in ref_cite_texts:
                ref = ref.replace(cite, "cite")
            new_refs.append(ref)
            
            pred_cite_texts = set(p.findall(pred))
            for cite in pred_cite_texts:
                pred = pred.replace(cite, "cite")
            new_preds.append(pred)
       
    else:
        new_preds = preds
        new_refs  = refs
            
    return new_preds, new_refs
    
def run_evaluate(args):
    
    references = get_references(args.dataset_name)
        
    metric = evaluate.load(args.metric_name)
    
    # predictions_file = glob(os.path.join(args.predictions_file, "*.txt"), recursive=True)
    
    predictions_file = args.predictions_file
    
    predictions_dir = os.path.dirname(predictions_file)
    
    results = {}
    
    with open(predictions_file) as f:
        predictions = f.readlines()
        predictions, references = post_process(args.dataset_name, predictions,references)
        
        results = metric.compute(predictions=predictions, references=references)
        
    with open(os.path.join(predictions_dir, "results_post.json"), "w") as f:
        json.dump(results,f, ensure_ascii=False)
            

def main():
    parser = ArgumentParser()

    parser.add_argument("--dataset_name",
                        type=str,
                        default=None)
    
    parser.add_argument("--task",
                        type=str,
                        choices=['summarization', 'science_summarization'],
                        default="summarization")

    parser.add_argument("--metric_name",
                        type=str,
                        default="rouge")
    
    parser.add_argument("--predictions_file",
                        type=str,
                        default="rouge")
    
    args = parser.parse_args()
    
    run_evaluate(args) 
    
if __name__ == "__main__":
    main()    