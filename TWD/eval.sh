list="1 2 3 4 5 6 7 8 9 10 "
dir="./pegasus-large-finetuning-512-b128-lr0.0001-gas8-science_summarization-desc-rand42/base/predictions"

for var in $list
do 
python eval.py  --task "science_summarization" --dataset_name "yaolu/multi_x_science_sum" --metric_name "rouge" --predictions_file $dir"/checkpoint-${var}/predictions.txt" 
done