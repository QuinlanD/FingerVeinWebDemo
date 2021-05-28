

FingerVeinDemo

————data  //存放测试用的数据

————dl   //深度学习的模块

————output // 

————web   //web后端





[toc]





## 如何运行webdemo：

anaconda切换到对应环境： conda acitvate stri



运行flask： flask run -h '0.0.0.0' -p '7897'

或者运行： runserver.py



## 规范

1. 图片命名规范：

应满足规范：[用户id]\_[lr]\_[imr]_[1-6].jpg

正确规范如：1_l_i_1.bmp    

错误示例：1_l_i_1_F.jpg， 错误示例将会导致用户注册错误，结果将一个用户当成6个用户。

1_l_i_1.bmp    ——》保存到数据库用户名 1_l_i

1_l_i_1_F.jpg  ——》保存到数据库用户名 1_l_i_1



2.神经网络权重应存储在 ./dl/checkpoint/<对应的神经网络类型>/<具体的网络权重文件>；对应的实现代码应存放在 ./dl/net/<对应的网络实现类>



## dl模块

这部分记录dl模块开发的思路

dl

————checkpoints //模型权重文件，用于载入模型

————database  //存放每个模型点对应的嵌入向量

————info

————loss    

————net   // 神经网络模型开发人员

config.py  // 配置信息

utils.py  // 不仅是指静脉深度学习模块的测试工具，还为web后端调用提供服务



utils.py向web提供的服务：

1.load_model 加载模型 

参数：模型保存点的路径

返回：加载后的模型



2.load_image 加载图片

参数：图片路径， torch.transform 预处理图片

返回：加载并处理后的图片



3.register 注册

参数： 加载的模型， 模型保存点名称， 图片路径

返回： 行号

该函数获取config内定义的注册路径，拼接模型保存点名称，创建一个templates.csv文件

templates.csv文件用于存放嵌入向量，每行一个。

该函数将图片嵌入向量保存到csv后，返回在templates.csv对应的行号



4.verif 1:1验证

参数：加载的模型， 图片1路径， 图片2路径

返回：两个图片的相似度



5.recognize 1:n识别

参数： 加载的模型， 图片路径， 模型保存点名称， 阈值

返回：一个列表，内容是在templates.csv中与图片的差距小于给定阈值的行号





6.analysis 统计

参数： 加载的模型， 模型保存点名称，数据集路径， 距离，torch.transform预处理器

返回：eer图片的路径， eer的值，阈值的范围， 



该函数需要加载指定的数据集路径，对图片做预处理，并用指定的距离来对图片处理——生成阈值范围【0.6，1】，每个阈值计算相应的far，frr，将结果保存到output/analysis文件中，根据时间戳创建文件，存储thresholds、fars、 frrs。



ps:由于用欧式距离计算速度较慢，先实现用余弦距离统计的方式



功能：

1.创建文件

2.生成图片

3.计算eer值

生成阈值，选择距离函数

加载数据



X 【X1， X2， 。。。 X6m】6m个向量

Y 【Y1, Y2,....Y6m】6m个标签

如何计算far和frr

余弦距离和欧式距离：



far误识率：将不同人当成同人

frr误拒率：将同人当成不同人

假设一个人有6张图片，共有m个人

对于每一个阈值：

far 计算： m * (5+4+3+2+1) = 15m

frr 计算： m * 6 * (6m - 6) /2 = 18m^2 -18m

有k段阈值：

总的计算需要 k * ( 18m^2 - 3m)

50 * ( 18 * 6000 * 6000 - 3 * 6000) = 32,399,100,000



计算far 时间复杂度 O（m）

计算frr 时间复杂度 O (m^2)



阈值划分为k段

总的时间复杂度 k* 36m^2



web中analysis：

返回图片、结果描述、将结果保存到数据库。

analysis数据库字段：

id;

picture_url;

result_url;

eer; 





## 如何扩展dl模块，添加深度学习模型



主要是依赖两个文件：config.py和util.py

config.py是一个配置文件，存在模型、数据等路径信息

util.py是一个向web提供服务的接口，需要实现以下几个函数：



```
load_model(checkpoint_path):

"""

: type checkpoint_path: str

: type device: str

: rtype: torh.nn.Module

"""

传入一个模型的路径checkpoint_path，将模型加载进内存，以指定的设备device(cpu或者gpu)运行，返回模型的引用
```



```
register(model, modelpoint, pic_path):

"""

: type model: torh.nn.Module

: type modelpoint: str

: type pic_path: str

: rtype: int

"""

用模型处理pic_path的图像，转换成嵌入向量，追加保存到以modelpoint为文件夹下的`templates.csv`文件中，并返回行号
```



```
verify(model, dist, img_path1, img_path2):

"""

: type model: torh.nn.Module

: type dist: str

: type img_path1: str

: type img_path2: str

: rtype: float

"""
将指定图像路径img_path1、img_path2通过model转换为嵌入向量，以指定的距离dist("Euclidean", "cos")，来计算并返回两个嵌入向量的分数
```



```
recognize(model, dist, pic_path, modelpoint):

"""

: type model: torh.nn.Module

: type dist: str

: type pic_path: str

: type modelpoint: str

: rtype: list --> [row1, row2 ... ] in template.csv

"""

用model将pic_path的图像转换为嵌入向量，加载modelpoint文件夹下的`templates.csv`文件（存放嵌入向量，每行放一个行号），以指定的距离dist("Euclidean", "cos")计算两者的距离，对比阈值eps，满足则将行号加入list，最后返回一个包含所有可能的行号的列表。
```



```
analysis(model, modelpoint, dataset):

"""

: type model: torh.nn.Module

: type modelpoint: str

: type dataset: str

: rtype: picture_url: str

: rtype: file_url: str

"""

加载数据集文件dataset，并用转换器trans对图像进行预处理，用模型model计算出far & frr的结果，并绘制好结果图像，放在picture_url，将对应的阈值、far、frr的值保存到file_url文件，返回结果
```





## 更加简单的扩展dl模块，添加深度学习模块

从以上的接口进一步抽象，提取dl部分的核心代码。两个步骤简单扩展：



1.需要**实现"FingerVeinDemo/dl/net/AbstractModel.py"的抽象接口**。

如：FingerVeinDemo/dl/net/bninception.py 中 定义了Net实现了AbstractModel抽象接口。



2.**修改配置文件：FingerVeinDemo/dl/config.py**中模块路径

```python
model_module = {
    "siamese": "dl.net.siamese",  # “类型”： “模块位置”
    "inception": "dl.net.bninception",
    "resnet": "dl.net.resnet18",
}
```

添加网络的类型和模块的位置。

将训练后的权重文件保存到FingerVeinDemo/dl/checkpoints下

创建神经网络的类型文件夹（如：FingerVeinDemo/dl/checkpoints/inception），将权重文件拷贝到该目录下





