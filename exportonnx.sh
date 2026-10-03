python export_onnx.py \
  --input checkpoint/20261003_1632_59266_Mobile_U_ViT/20261003_1632_59266_Mobile_U_ViT_best.pth \
  --output model_best_opset16.onnx \
  --model Mobile_U_ViT \
  --img_size 256 \
  --num_classes 1 \
  --opset 16