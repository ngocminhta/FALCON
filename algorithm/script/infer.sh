Model_PATH="pth/Deepfake_best.pth" #PATH for the model ckpt
DATABASE_PATH="database/deepfake" #PATH for the database
python infer.py --database_path ${DATABASE_PATH} --model_path ${Model_PATH} --K 5\
    --text "I really want someone to change my view on this, since everyone I know are frowning on me for thinking this way. My argument is, that just with my single vote wouldn't have any effect in the result and thus, it's not worth voting at all But if you don't vote then your opinion doesn't count"