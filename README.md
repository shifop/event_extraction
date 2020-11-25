# event_extraction
科大讯飞2020事件抽取挑战赛第九名方案

>这份是提交给比赛举办方的代码，训练代码以及建模思路后续补充（看心情）

## 项目结果说明

data: 存储训练数据

model: 存储模型文件

sub: 存储提交文件

predict_trigger.py: 预测触发词

predict_event.py：预测其他论元

process.py : 融合多个模型的预测结果

python predict_tense_polarity.py: 预测极态和时态并生成最终提交文件

model_mcr.py： 预测触发词和其他论元的模型的结构

model_dt_v2.py： 预测极态和时态的模型的结构

## 运行说明

依次运行：

python predict_trigger.py

python predict_event.py

python process.py

python predict_tense_polarity.py

最终预测结果保持在./sub/sub.json文件夹

## 环境要求

tensorflow-gpu==2.1.0

bert-for-tf2==0.14.6

tqdm

