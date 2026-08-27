python mainStage80.py --weights pretrained weights/ntu 60/joint/runs-128-80128.pt --phase test --save-score True --config config/nturgbd-cross-subject/joint.yaml --device 0 --start-epoch 128

python mainStage80.py --weights pretrained weights/ntu 120/joint/runs-124-122016.pt --phase test --save-score True --config config/nturgbd120-cross-subject/joint.yaml --device 0 --start-epoch 124
