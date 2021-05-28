from abc import ABC, abstractclassmethod


class AbstractModel(ABC):

    @abstractclassmethod
    def loadModel(self, model_path):
        """
        : type model_path: str
        : rtype: torh.nn.Module
        完成神经网络从指定路径`model_path`的加载
        """
        pass

    @abstractclassmethod
    def getDataLoader(self, dataset):
        """
        : type model_path: str
        : rtype: torch.utils.data.DataLoader
        返回一个加载文件夹路径dataset用于测试的dataloader。
        """
        pass

    @abstractclassmethod
    def getOutput(self, img_path):
        """
        : type img_path: str
        : rtype: torch.Tensor
        根据图片路径img_path进行处理，返回一个最终用于存储、匹配的嵌入向量。
        """
        pass

    @abstractclassmethod
    def getOutputByVec(self, input_vec):
        """
        : type input_vec: str
        : rtype: torch.Tensor
        input_vec是由DataLoader获取的数据输入（不含标签），根据输入数据返回一个最终用于存储、匹配的嵌入向量。
        """
        pass

    @abstractclassmethod
    def getThreshould(self, dist):
        """
        : type dist: str
        : rtype: float
        根据dist对应的距离计算标准，获取阈值
        example:
            dist = "Euclidean"
            return 0.008
            
            dist = "cos"
            retrun 0.001
        """
        pass

    @abstractclassmethod
    def getThreshouldRange(self):
        """
        : rtype: tuple (threshould_low_bound, threshould_high_bound, iter_num)
        获取最后计算far & frr时，迭代的阈值范围，和迭代次数。
        example: 
            return 0.6, 0.9, 50
            将0.6到0.9之间分成50个阈值，迭代每个阈值下far和frr。
        """
        pass