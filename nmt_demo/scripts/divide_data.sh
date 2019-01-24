python src/split_file.py ./data/datasets/$1/pair_data.from ./data/datasets/$1/train.from 0.0 0.8
python src/split_file.py ./data/datasets/$1/pair_data.from ./data/datasets/$1/vali.from 0.8 0.9
python src/split_file.py ./data/datasets/$1/pair_data.from ./data/datasets/$1/test.from 0.9 1.0

python src/split_file.py ./data/datasets/$1/pair_data.to ./data/datasets/$1/train.to 0.0 0.8
python src/split_file.py ./data/datasets/$1/pair_data.to ./data/datasets/$1/vali.to 0.8 0.9
python src/split_file.py ./data/datasets/$1/pair_data.to ./data/datasets/$1/test.to 0.9 1.0


