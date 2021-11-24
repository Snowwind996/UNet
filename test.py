import json
from data import imageread
def jsonread(filename):
    with open(filename, 'r') as file:
        js = file.read()
        dic = json.loads(js)
        print(dic)
        file.close()
    return dic
def txtread(filename):
    with open(filename, 'r+') as fr:
        dic = eval(fr.read())  # 读取的str转换为字典
        print(dic)
        fr.close()
    return dic
filename='resout1.txt'
# jsonread(filename)
def tryit():
    import numpy as np
    x = np.random.randint(8,size=9).reshape(3,3)
    # label=np.where(x == x.max())
    # print(x)
    # print(label)
    # print(x[label])
    print(f"x is:{x}")
    ret=x.argsort()
    # print(np.array(x))
    print(f"ret is :{ret}")
    print(f"x[ret]is :{x[ret]}")

def trydataset():
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    imgset,goalset=imageread(imgnum=50,figureshow='off')
    train_ids = TensorDataset(imgset, goalset)
    train_loader = DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
    for i, data in enumerate(train_loader, 1):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
        x_data, label = data
        print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))
        break
    print("封装成功！")
def try33():
    import time
    time_now=time.strftime('%Y-%m-%d %H:%M:%S')
    print(".\\"+time_now+"result\\"+str(1)+"img.png")


import numpy as np
import shapely
from shapely.errors import TopologicalError
from shapely.geometry import Polygon, MultiPoint

class cal_iou():
    def calculate_iou(self,actual_quadrilateral, predict_quadrilateral):
        """

        :param actual_quadrilateral: 预测四边形四个点坐标的一维数组表示，[x,y,x,y....]
        :param predict_quadrilateral: 期望四边形四个点坐标的一维数组表示，[x,y,x,y....]
        :return:
        """

        def to_polygon(quadrilateral):
            """

            :param quadrilateral: 四边形四个点坐标的一维数组表示，[x,y,x,y....]
            :return: 四边形二维数组, Polygon四边形对象
            """
            # 四边形二维数组表示
            quadrilateral_array = np.array(quadrilateral).reshape(4, 2)
            # Polygon四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
            quadrilateral_polygon = Polygon(quadrilateral_array).convex_hull

            return quadrilateral_array, quadrilateral_polygon
        # 预测四边形二维数组, 预测四边形 Polygon 对象
        actual_quadrilateral_array, actual_quadrilateral_polygon = to_polygon(actual_quadrilateral)
        # 期望四边形二维数组, 期望四边形 Polygon 对象
        predict_quadrilateral_array, predict_quadrilateral_polygon = to_polygon(predict_quadrilateral)
        # 合并两个box坐标，变为8*2 便于后面计算并集面积
        union_poly = np.concatenate((actual_quadrilateral_array, predict_quadrilateral_array))
        # 两两四边形是否存在交集
        inter_status = actual_quadrilateral_polygon.intersects(predict_quadrilateral_polygon)
        # 如果两四边形相交，则进iou计算
        if inter_status:
            try:
                # 交集面积
                inter_area = actual_quadrilateral_polygon.intersection(predict_quadrilateral_polygon).area
                # 并集面积 计算方式一
                # union_area = poly1.area + poly2.area - inter_area
                # 并集面积 计算方式二
                union_area = MultiPoint(union_poly).convex_hull.area
                # 若并集面积等于0,则iou = 0
                if union_area == 0:
                    iou = 0
                else:
                    # 第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
                    iou = float(inter_area) / union_area
                    #  第二种： 交集 / 并集（常见矩形框IOU计算方式）
                    # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            except shapely.errors.TopologicalError:
                print('shapely.errors.TopologicalError occured, iou set to 0')
                iou = 0
        else:
            iou = 0

        return iou

    def __init__(self, actual_quadrilateral = [908, 215, 934, 312, 752, 355, 728, 252],predict_quadrilateral = [923, 308, 758, 342, 741, 262, 907, 228]):
        self.actual_quadrilateral=actual_quadrilateral
        self.predict_quadrilateral=predict_quadrilateral
        iou = self.calculate_iou(self.actual_quadrilateral, self.predict_quadrilateral)
        print(iou)



if __name__=="__main__":
    cal_iou()
