# 命名实体识别

改自该项目: https://github.com/luopeixiang/named_entity_recognition.git

改动如下：
+ 将原来同时训练模型改为可以单个训练或者测试；
+ 原有代码冗余较多，将模型写成工厂模式，减少代码冗余；
+ 增加预测函数，可以基于得到的模型文件预测新来的文本；

##快速开始

1. （推荐）运用conda或者其他工具，创建conda虚拟环境，与系统的环境隔离。

安装依赖项
```shell script
pip3 install -r requirement
```

2. 训练和测试遵循以下格式

```shell script
python main.py action --model Model --dataset Dataset
```
其中参数的含义如下
action为操作， 训练或者测试,[train test]
Model为已实现的模型，可填写范围为[HMM, CRF, BILSTM, BILSTM_CRF],
Dataset为数据集，用户可自行扩展数据集，以原作者的数据集为例ResumeNER为例，

训练BILSTM模型
```shell script
python main.py train --model BILSTM --dataset ResumeNER
```

测试BILSTM模型在ResumeNER数据集上的效果，输出准确率，召回率和混淆矩阵
```shell script
python main.py test --model BILSTM --dataset ResumeNER
```
3. 用模型来预测新的文本
```shell script
python predict --model Model --dataset Dataset --text Text
```
Model和Dataset参数含义同上，text为要预测的文本。

以BILSTM_CRF模型预测一段话，如下，
```shell script
python predict.py --model BILSTM_CRF --dataset ResumeNER --text "钱学森，中国科学家院士"
```

## 模型介绍
请参看原项目作者的介绍，内容在README_old.md文件中。