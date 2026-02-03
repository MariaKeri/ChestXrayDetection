Dáta XRay: https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
Dáta COVID: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Projekt obsahuje dva tréningové skripty:
 - Train_XRay.py  (tréning na XRay Pneumonia datasete)
 - Train_COVID.py (tréning na COVID Radiography datasete)

Spustenie:
  python3 Train.py
  python3 Train_COVID.py

Očakávané umiestnenie datasetov:
  /content/data/chest_xray
  /content/data/COVID-19_Radiography_Dataset

Každý skript automaticky:
  - vytvorí stratifikovaný split 70/15/15 (splits.csv)
  - trénuje Inception-v3 v dvoch fázach (feature extraction + fine-tuning)
  - používa class weights, early stopping (val AUC), LR scheduler
  - uloží best_model.pt podľa validation AUC
  - vyhodnotí test set (Accuracy, Sensitivity, Specificity, AUC)
  - vygeneruje ROC krivku, kalibračnú krivku, Brier score
  - vytvorí Grad-CAM galériu
  - uloží MODEL_CARD.md

Výstupy sa ukladajú do priečinka:
  run_<DATASET>_<timestamp>
 \

