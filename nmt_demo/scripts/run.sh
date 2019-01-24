#python src/total_preprocess.py ./data
python src/find_best_pair.py ./data/process $1 $2
mkdir -p ./data/datasets/$3
python src/make_dataset.py ./data/process/ ./data/datasets/$3/ $1 $2
. scripts/divide_data.sh $3
. scripts/train.sh $3

