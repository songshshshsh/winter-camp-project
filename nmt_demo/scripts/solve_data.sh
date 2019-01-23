#python src/total_preprocess.py ./data
#python src/find_best_pair.py ./data/process $1 $2
#mkdir -p ./data/datasets/$12$2
#python src/make_dataset.py ./data/process/ ./data/datasets/$12$2/ $1 $2
. scripts/divide_data.sh $1 $2
. scripts/train.sh $1 $2

