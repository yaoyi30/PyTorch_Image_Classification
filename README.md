<div align="center">   

# PyTorch Image Classification Project
</div>

### 环境配置
python version 3.8, torch 1.8.1, torchvision 0.9.1:
```
pip install torch==1.8.1 torchvision==0.9.1
```


### 数据准备
数据文件夹结构如下:
```
datasets/
  train/   # train images
     class_1/
        img1.jpg
        img2.jpg
         .
         .
         .
     class_2/
        .
        .
        .
  val/     # val images
     class_1/
        img1.jpg
        img2.jpg
         .
         .
         .
     class_2/
        .
        .
        .

```
### 训练
```
python train.py --input_size 224 224 --batch_size 32 --epochs 100 --nb_classes 10 --data_path ./datasets/ --output_dir ./output_dir 
```
### 评价模型
```
python eval.py --input_size 224 224 --batch_size 8 --weights ./output_dir/best.pth --data_path ./datasets/ --nb_classes 10
```
### 模型预测
```
python predict.py --input_size 224 224 --weights ./output_dir/best.pth --image_path ./1.jpg --nb_classes 10
```
### 导出onnx模型
```
python export_onnx.py --input_size 224 224 --weights ./output_dir/best.pth --nb_classes 10
python -m onnxsim best.onnx best_sim.onnx
```

### 结果可视化
#### 1. Accuracy曲线
![acc.png](output_dir%2Facc.png)
#### 2. Loss曲线
![loss.png](output_dir%2Floss.png)
#### 3. 学习率曲线
![learning_rate.png](output_dir%2Flearning_rate.png)
#### 4. 混淆矩阵
![confusion_matrix.png](output_dir%2Fconfusion_matrix.png)
#### 5. onnx模型结构(简化后)
![onnx.png](output_dir%2Fonnx.png)

### 代码分析
https://blog.csdn.net/qq_38412266/article/details/139047128?spm=1001.2014.3001.5501