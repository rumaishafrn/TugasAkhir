# Notebook

Folder ini disediakan untuk notebook eksplorasi dan reproduksi hasil.

Pelatihan pada penelitian asli dijalankan di **Kaggle** dengan GPU Tesla T4.
Kerangka notebook untuk Kaggle atau Colab:

```python
# Sel 1
!git clone https://github.com/USERNAME/volumetric-estimation-rgb.git
%cd volumetric-estimation-rgb
!pip install -q ultralytics transformers accelerate timm

# Sel 2: pelatihan
!python scripts/01_train_yolo.py --data /kaggle/input/dataset/data.yaml --all-variants

# Sel 3: estimasi volume
!python scripts/03_run_waste_pipeline.py \
    --weights runs/segment/waste-yolov8s-seg/weights/best.pt \
    --video /kaggle/input/dataset/uji.mp4 \
    --ground-truth data/ground_truth_waste.csv

# Sel 4: perbandingan seluruh model MDE
!python scripts/05_compare_mde_models.py --case waste ...
```
