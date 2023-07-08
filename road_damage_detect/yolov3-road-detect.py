# %% [markdown]
# # YOLOV3实现道路路面病害检测分析-paddle代码学习解析

# %% [markdown]
# # 一、项目说明
# 本项目是用来学习飞浆框架下深度学习的实现方法，项目选用yolov3算法应用作为学习对象。项目目标目的是本着将数据处理、模型算法、模型输出、端到端布置过程都源代码实现，并提供注释说明（代码内注释说明更详细点），以便更多人能够较为全面、深入了解深度学习算法是如何工作的。由于本人对飞浆框架及众多算法理解不够，项目内容只能陆续完善，个人也是在不断的学习过程中，感兴趣的也可以自建项目完善说明。
# 项目主要学习借鉴了如下课程以及相关学员的帮助指导，在此予以感谢：
# * [AI识虫比赛](https://aistudio.baidu.com/aistudio/projectdetail/3423143)
# * [百度架构师手把手带你零基础实践深度学习](https://aistudio.baidu.com/aistudio/education/group/info/1297)
# * [【全球开放数据创新应用大赛】道路路面病害智能分析baseline](https://aistudio.baidu.com/aistudio/projectdetail/2037359?channelType=0&channel=0) 
# * 还有感谢飞浆资深导师毕然、讲目标检测yolov3的导师、AI达人创造营第二期班长、助教等人。
# * 特别还要说明一下，paddlex提供了非常方便的使用工具，更多的模型训练、应用部署都可推荐使用，本学习项目仅只是为了认识学习paddle环境下模型实现方法，以便于在paddlex使用时更好了灵活使用相应工具。

# %% [markdown]
# # 二、数据说明
# * 本项目选用数据为【全球开放数据创新应用大赛】道路路面病害智能分析所提供的数据，使用者可以自行挂载相应数据。
# * 数据提供车载摄像头拍摄数据，共14000张道路病害图像样本，其中训练集提供标注标签（病害类别及目标框位置），测试集不提供标注标签。
# * 图像数据为三通道JPG图像，尺寸为1600×1184，标签COCO格式的json文件，使用utf-8编码。训练集6000张图片，测试集A榜2000张图片，测试集B榜6000张图片。
# * 对所挂数据进行解压缩，并放到指定文件夹下，具体执行如下代码

# %%
# Way_One:
# !unzip -oq /workspaces/roadai/dataset/全球开放数据应用创新大赛数据集/train.zip -d dataset
# !unzip -oq /workspaces/roadai/dataset/全球开放数据应用创新大赛数据集/test_A.zip -d dataset

# %% [markdown]
# # 三、数据处理

# %% [markdown]
# ### trainsiton.py
# 1. 主要实现将解压缩后的数据集生成标签文件和图片，路径根据实际情况自己修改
# 2. 需要安装相应的依赖库，包括cocoapi、pycocotools、lxml
# 3. 在dataset文件夹下建立VOC文件夹，可以在左侧手动建立文件夹，也可以通过如下代码建立文件夹，生成VOC文件夹后要将路径修改为默认根路径
# 4. 接下来就可以执行transition.py了，执行时要注意查看对应路径是否正确。代码会执行生成标签文件和图片。

# %%
# !git clone git@github.com:cocodataset/cocoapi.git
# !pip install pycocotools -i https://mirror.baidu.com/pypi/simple
# %cd cocoapi/PythonAPI/
# !make
# !python setup.py install
# !pip install lxml

# %%
# %cd dataset/
# !mkdir VOC
# %cd /home/aistudio/

# %%
!python transition.py

# %%


# %% [markdown]
# ### insects_reader.py
# * 主要实现读取图片路径信息，标签信息等
# * 文件可以独立执行，执行结果是打印出来records记录中的第八个数据，信息包括图片路径、id、长、宽、分类标签gt_calss、标识框gt_bbox等等
# * 注意records记录是打乱顺序的，所以records[8]并不代表第八张图片的信息。

# %%
!python insects_reader.py

# %% [markdown]
# ### reader.py
# * reader.py引用了insects_reader.py、image_utils.py文件。
# * 其中insects_reader.py是读取图像、标签等信息，如上所述；
# * image_utils.py则是图像增广处理，具体请看源代码中注释，通常用户可以根据自己项目添加设置图像预处理方式。
# * image_utils.py文件中引用了box_utils.py文件，该文件有三个函数，分别为计算xyxy形式框的IOU、计算xywh形式框的IOU、批量计算IOU、。。。
# * 文件可以单独执行，输出img、gt_box、gt_lables的数据格式

# %%
# !export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/NsightSystems-cli-2023.2.3/host-linux-x64/

# %%
!python reader.py

# %% [markdown]
# ### draw_anchors.py
# draw_anchors.py是实现在图片上标注锚框，提供了测试图片，感兴趣的可以调整数据参数后再执行文件，便于直观理解。

# %%
!python draw_anchors.py

# %% [markdown]
# ### anchor_lables.py
# * 前面的draw_anchors.py是给了一个一张图片，固定设置锚框大小尺寸，然后打印出来，而anchor_lables.py则是调用图像集和标签信息来计算锚框的标签信息。
# * 注意我的平台不能用multithread_loader，只能使用data_loader来读取加载图片信息。
# * 执行文件会打印出来所有图片的数据结构。都标注出了objectness为正的预测框，剩下的预测框则默认objectness为0，对于objectness为1的预测框，标出了他们所包含的物体类别，以及位置回归的目标，scale_location用来调节不同尺寸的锚框对损失函数的贡献，作为加权系数和位置损失函数相乘。
# * 给出的应该是p0层的锚框输出。

# %%
!python anchor_lables.py

# %% [markdown]
# ### 锚框大小选择和图像均值参数选择问题
# * 可能很多人在执行yolov3模型时，都选用昆虫项目案例中的mean = [0.485, 0.456, 0.406]，std = [0.229, 0.224, 0.225]，如下代码给出了均值和方差选择的一种方法；
# * 锚框大小选择问题，目前我依然不太清楚怎么处理。在此前项目执行时采用昆虫项目给出的锚框，此前训练200轮，损失函数最终只到4左右，模型输出预测，还比较准。感兴趣的可以执行试试看，模型为yolo_epoch199.pdparams。
# * 代码给出的锚框如下，三层分别使用对应的锚框大小。ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]，ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],每层3个锚框。
# * 飞浆成员“白鱼”在其路面情况检测项目中，给出的锚框选择貌似是[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]，五组锚框，但是不知是哪一层的？

# %%
import glob
import cv2
import numpy as np
import tqdm

def get_mean_std(image_path_list):
    print('Total images:', len(image_path_list))
    max_val, min_val = np.zeros(3), np.ones(3) * 255
    mean, std = np.zeros(3), np.zeros(3)
    for image_path in tqdm.tqdm( image_path_list):
        image = cv2.imread(image_path)
        for c in range(3):
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
            max_val[c] = max(max_val[c], image[:, :, c].max())
            min_val[c] = min(min_val[c], image[:, :, c].min())

    mean /= len(image_path_list)
    std /= len(image_path_list)

    mean /= max_val - min_val
    std /= max_val - min_val

    return mean, std


mean, std = get_mean_std(glob.glob('./dataset/VOC/JPEGImages/*.jpg'))
print('mean:', mean)
print('std:', std)

# %% [markdown]
# # 四、模型选择
# YOLOv3算法的基本思想可以分成两部分：
# * 按一定规则在图片上产生一系列的候选区域，然后根据这些候选区域与图片上物体真实框之间的位置关系对候选区域进行标注。跟真实框足够接近的那些候选区域会被标注为正样本，同时将真实框的位置作为正样本的位置目标。偏离真实框较大的那些候选区域则会被标注为负样本，负样本不需要预测位置或者类别。
# * 使用卷积神经网络提取图片特征并对候选区域的位置和类别进行预测。这样每个预测框就可以看成是一个样本，根据真实框相对它的位置和类别进行了标注而获得标签值，通过网络模型预测其位置和类别，将网络预测值和标签值进行比较，就可以建立起损失函数。
# ![](https://ai-studio-static-online.cdn.bcebos.com/0f64b7c6e82445849b4f81bc77f4b11551f4a22209fd4af4b2858fbad9647b5f)
# 
# * 左边是输入图片，上半部分所示的过程是使用卷积神经网络对图片提取特征，随着网络不断向前传播，特征图的尺寸越来越小，每个像素点会代表更加抽象的特征模式，直到输出特征图，其尺寸减小为原图的1/32。
# * 下半部分描述了生成候选区域的过程，首先将原图划分成多个小方块，每个小方块的大小是32×32，然后以每个小方块为中心分别生成一系列锚框，整张图片都会被锚框覆盖到。在每个锚框的基础上产生一个与之对应的预测框，根据锚框和预测框与图片上物体真实框之间的位置关系，对这些预测框进行标注。
# * 将上方支路中输出的特征图与下方支路中产生的预测框标签建立关联，创建损失函数，开启端到端的训练过程。
# 

# %% [markdown]
# ### yolov3.py
# * YOLOv3(paddle.nn.Layer)类，初始化的时候，道路检测的类别是8个，这个地方对num_classes进行了修改；
# * 飞桨给出的计算p0、p1、p2层的损失函数的函数为paddle.vision.ops.yolo_loss；
# * 为便于理解yolov算法，提供了p0层的损失函数计算并打印看看运行效果。

# %%
!python yolov3.py

# %% [markdown]
# ### train.py
# * 可以使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数，实现多进程读取数据，但试了本地CPU环境不能用。
# * 也可以使用自带data_loader函数读取数据，设置batchsize，但只能单进程执行，暂不能执行多进程multithread_loader，原因是自己对相应调用的函数还没理解。
# 使用paddle自带的paddle.vision.ops.yolo_loss，直接计算p0、p1、p2层损失函数，过程更简洁，速度也更快

# %%
!python train.py

# %% [markdown]
# ### 模型导出inference
# * 模型导出采用paddle.jit.save保存为静态模式，执行的结果我们可以看出生成了inference文件夹，里面保存了存储的模型结构 Program 文件的后缀为 .pdmodel ，存储的持久参数变量文件的后缀为 .pdiparams ，同时这里也会将一些变量描述信息存储至文件，文件后缀为 .pdiparams.info。
# * paddle2.0之后也提供了save_inference_model进行动静转化

# %%
# save inference model
import paddle
from yolov3 import YOLOv3
from paddle.static import InputSpec
#
model = YOLOv3(num_classes=8)
# 加载训练好的模型参数
state_dict = paddle.load("./yolo_epoch199.pdparams")
# 将训练好的参数读取到网络中
model.set_state_dict(state_dict)
# 设置模型为评估模式
model.eval()

# 保存inference模型
paddle.jit.save(
    layer=model,
    path="inference/model",
    input_spec=[InputSpec(shape=[1, 3, 640, 640], dtype='float32')]
)

print("==>Inference model saved in inference")

# %% [markdown]
# # 五、模型评估

# %% [markdown]
# ### predict_all.py
# * 下面是完整的测试程序，在测试数据集上的输出结果将会被保存在pred_results.json文件中。
# * 预测框列表中每个元素[label, score, x1, y1, x2, y2]描述了一个预测框，label是预测框所属类别标签，score是预测框的得分；x1, y1, x2, y2对应预测框左上角坐标(x1, y1)，右下角坐标(x2, y2)。每张图片可能有很多个预测框，则将其全部放在预测框列表中。

# %%
!python predict_all.py

# %% [markdown]
# ### predict.py
# * 用保存的yolo_epoch199.pdparams模型进行预测效果，图片选用dataset/test_A/images/00100.jpg，输出为output_pic.png，我们可以看到模型识别并标注了井盖、指示箭头图框。
# * 该模型经过200轮训练，能够在置信度30%以上检测出多个特征边框。
# * 由于对yolov3还没有吃透，目前检测出来是在mean、anchors等方面可以改，但也不知道效果如何。

# %%
! python predict.py

# %% [markdown]
# # 六、后期规划
# * 　模型参数需要优化；　　
# * 	端到端部署流程熟悉应用；
# * 　yolov3里面get_loss_self等函数消化实现。。。


