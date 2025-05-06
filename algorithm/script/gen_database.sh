DATA_PATH="/opt/LLM_detect_data" #PATH for the data
Model_PATH="/opt/DeTeCtive/pth/Deepfake_best.pth" #PATH for the model ckpt

# deepfake
python gen_database.py --device_num 8 --batch_size 128 --model_name princeton-nlp/unsup-simcse-roberta-base \
                   --mode deepfake --database_path ${DATA_PATH}/Deepfake/cross_domains_cross_models --database_name 'train' \
                   --model_path ${Model_PATH} --save_path database/deepfake

# # TuringBench
# python gen_database.py --device_num 8 --batch_size 128 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode Turing --database_path ${DATA_PATH}/TuringBench/AA --database_name 'train' \
#                    --model_path ${Model_PATH} --save_path database/TuringBench

# # M4-monolingual
# python gen_database.py --device_num 8 --batch_size 128 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode M4 --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'monolingual_train' \
#                    --model_path ${Model_PATH} --save_path database/M4-monolingual

# # M4-multilingual
# python gen_database.py --device_num 8 --batch_size 128 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode M4 --database_path ${DATA_PATH}/SemEval2024-M4/SubtaskA --database_name 'multilingual_train' \
#                    --model_path ${Model_PATH} --save_path database/M4-multilingual

# # OUTFOX
# python gen_database.py --device_num 8 --batch_size 128 --model_name princeton-nlp/unsup-simcse-roberta-base \
#                    --mode OUTFOX --database_path ${DATA_PATH}/OUTFOX --database_name 'train' \
#                    --model_path ${Model_PATH} --save_path database/OUTFOX