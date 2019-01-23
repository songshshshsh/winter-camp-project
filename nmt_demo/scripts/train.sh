mkdir -p ./model_sets/$12$2_attention_model/
python -m nmt.nmt.nmt \
	        --src=from --tgt=to \
		    --vocab_prefix=./data/datasets/$12$2/vo  \
		        --train_prefix=./data/datasets/$12$2/train \
			    --dev_prefix=./data/datasets/$12$2/vali  \
			        --test_prefix=./data/datasets/$12$2/test \
				    --out_dir=./model_sets/$12$2__attention_model \
				        --num_train_steps=1200 \
					    --steps_per_stats=100 \
					        --num_layers=2 \
						    --num_units=128 \
						        --dropout=0.2 \
							    --metrics=bleu


