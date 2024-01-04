
# 模式识别与机器学习大作业

作者: 刘鸣霄
邮箱: mx-liu21@mails.tsinghua.edu.cn

## 文件结构

- dataloader.py 数据集加载器
- model.py 模型
- train.py 训练
- evaluate.py 评估
- yolo_utils.py, dataloader_utils.py 辅助函数
- data_augmentation.py 数据增强
- unit_test.ipynb 单元测试
- test_on_my_own_img.ipynb 在自己的图片上测试
- README.md 本文件
- constants.py 常量
- marco.sh 向服务器上传和下载文件的脚本
- checkpoint/ 模型保存的文件夹
- Annotations/ 标注文件夹
- JPEGImages/ 图片文件夹
- logs/ 日志文件夹
- my_images/ 自己的图片文件夹
- results/ 评估结果文件夹

## 使用

### 如何训练:   
> python train.py | tee logs/*.txt  

我没有实现那种命令行参数的功能，所以只能通过修改config来修改参数  

注意要放好数据集  

### 快速体验:  

在my_images文件夹中放入自己的图片，然后运行test_on_my_own_img.ipynb即可  

下载我的权重weights220.pth:  

> https://cloud.tsinghua.edu.cn/seafhttp/files/5dcc1205-4d50-4aba-8d67-30c4fad68171/weights220.pth  

