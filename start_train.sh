cd /project/train/src_repo/v5/yolov5

echo "Prepare environment..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

echo "Processing data..."
python ../preprocess.py

echo "Start training..."
python train.py --batch-size 64 --epochs 300  --data ./data/cvmart.yaml --hyp ./data/hyps/hyp.scratch-low.yaml --weight ../yolov5s.pt --img 480 --project /project/train/models/ --cfg ./models/yolov5s.yaml
