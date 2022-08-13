cd /project/train/src_repo/yolov5

echo "Prepare environment..."
pip install -r requirements.txt

echo "Processing data..."
python preprocess.py

echo "Start training..."
python train.py --batch-size 64 --epochs 300  --data ./data/cvmart.yaml --hyp ./data/hyps/hyp.scratch.yaml --weight /project/train/src_repo/yolov5s.pt --img 480 --project /project/train/models/ --cfg ./models/yolov5s.yaml