data Xray: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
data COVID: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
na spustenie: python3 Train.py \
  --data_dir "cesta_k_datasetu/chest_xray" / "cesta_k_datasetu/COVID-19_Radiography_Dataset"\
  --batch_size 16 \
  --num_epochs 10 \
  --freeze_backbone
