#python src/total_preprocess.py ./data
#python src/find_best_pair.py ./data/process INTJ INFJ
mkdir -p ./data/datasets/INTJ2INFJ
python src/make_dataset.py ./data/process/ ./data/datasets/INTJ2INFJ/ INTJ INFJ

