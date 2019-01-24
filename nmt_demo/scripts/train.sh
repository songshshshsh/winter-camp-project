mkdir -p ./model_sets/$1_model/
python -m nmt.nmt \
	        --src=from --tgt=to \
		    --vocab_prefix=./data/datasets/$1/vo  \
		        --train_prefix=./data/datasets/$1/train \
			    --dev_prefix=./data/datasets/$1/vali  \
			        --test_prefix=./data/datasets/$1/test \
				    --out_dir=./model_sets/$1_model \
				        --num_train_steps=10000 \
					    --steps_per_stats=100 \
					        --num_layers=2 \
						    --num_units=128 \
						        --dropout=0.2 \
							    --metrics=bleu


