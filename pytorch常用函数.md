# pytorch常用函数

## 1. 非常常用

### 1.1 torch.nn.Embedding()

```python
class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
```

一个保存了固定字典和大小的简单查找表。

这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。

**参数：**

- **num_embeddings** (*[int](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)*) - 嵌入字典的大小
- **embedding_dim** (*[int](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)*) - 每个嵌入向量的大小
- **padding_idx** (*[int](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/), optional*) - 如果提供的话，输出遇到此下标时用零填充
- **max_norm** (*[float](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/), optional*) - 如果提供的话，会重新归一化词嵌入，使它们的范数小于提供的值
- **norm_type** (*[float](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/), optional*) - 对于max_norm选项计算p范数时的p
- **scale_grad_by_freq** (*boolean, optional*) - 如果提供的话，会根据字典中单词频率缩放梯度

**变量：**

- **weight (\*[Tensor](http://pytorch.org/docs/tensors.html#torch.Tensor)\*)** -形状为(num_embeddings, embedding_dim)的模块中可学习的权值

**形状：**

- **输入：** LongTensor *(N, W)*, N = mini-batch, W = 每个mini-batch中提取的下标数
- **输出：** *(N, W, embedding_dim)*

[pytorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)

根据库中的注释

```
Args:
    num_embeddings (int): size of the dictionary of embeddings
    embedding_dim (int): the size of each embedding vector
    padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the 									 gradient;
                                 therefore, the embedding vector at :attr:`padding_idx` is not updated 									 	 during training,
                                 i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                 the embedding vector at :attr:`padding_idx` will default to all zeros,
                                 but can be updated to another value to be used as the padding vector.
```

ps:

- i.e. 表示**也就是说，换句话说**的意思
- e.g. 表示**举个例子**
- etc. 表示等等

> http://www.360doc.com/content/20/1204/16/72757311_949490538.shtml
>
> 

### 1.2.torch.nn.MSELoss(size_average=True)

创建一个衡量输入`x`(`模型预测输出`)和目标`y`之间均方误差标准。

- 如果在创建`MSELoss`实例的时候在构造函数中传入`size_average=False`，那么求出来的平方和将不会除以`n`
- 这是一个类，一般要用它创建一个对象

```
criterion = torch.nn.MSELoss(size_average=False)
```


$$
loss(x,y)=1/n\sum(x-y)^2
$$



### 1.3 torch.nn.BCELoss(weight=None,size_average=True)

计算target与output之间的二进制交叉熵。



### 1.4 torch.optim.SGD(model.parameter(), lr=0.01)

- class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
- params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
- lr (`float`) – 学习率
- momentum (`float`, 可选) – 动量因子（默认：0）
- weight_decay (`float`, 可选) – 权重衰减（L2惩罚）（默认：0）
- 这也是一个类，用于创建对象

```
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()
```



###  1.5 Training Cycle

```
for epoch in range(100):
	y_pred = model(x_data)
	loss = criterion(y_pred,y_data)
	print(epoch,loss)  # loss是一个对象，打印时自动调用对象内的__str__()方法
	optimizer.zero_grad()  # 所有权重的梯度归零
	loss.backward()  # 反向传播
	ooptimizer.step()  # 更新权重，根据参数梯度和学习率进行更新
```



###  1.6 torch.nn.Linear(in_features, out_features, bias=True)

- class torch.nn.Linear(in_features, out_features, bias=True)
- 对输入数据做线性变换y=Ax+b

- **in_features** - 每个输入样本的大小
- **out_features** - 每个输出样本的大小
- **bias** - 若设置为False，这层不会学习偏置。默认值：True
- 输入（N，in_features）
- 输出（N，out_features）
- **weight** -形状为(out_features x in_features)的模块中可学习的权值
- **bias** -形状为(out_features)的模块中可学习的偏置

示例

```
>>> m = nn.Linear(20, 30)
>>> input = autograd.Variable(torch.randn(128, 20))
>>> output = m(input)
>>> print(output.size())
```



### 1.7 torch.nn.Conv2d(in_channels,out_channels,kernel_size)

- ###### class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

- 其参数至少要指定输入输出的channels数量和卷积核尺寸，输出channels数量为卷积核个数。

- in_channels(`int`) – 输入信号的通道

- out_channels(`int`) – 卷积产生的通道

- kerner_size(`int` or `tuple`) - 卷积核的尺寸

- stride(`int` or `tuple`, `optional`) - 卷积步长

- padding(`int` or `tuple`, `optional`) - 输入的每一条边**补充0的层数**

- bias(`bool`, `optional`) - 如果`bias=True`，添加偏置

- **input: (N,C_in,H_in,W_in)**
  **output: (N,C_out,H_out,W_out)，C_in和C_out分别指的是channels的数量，N指的是batch包含的样本数，H,W指的是长和宽。**

- **weight(`tensor`) - 卷积的权重，大小是(`out_channels`, `in_channels`,`kernel_size`)**
  **bias(`tensor`) - 卷积的偏置系数，大小是（`out_channel`）**

示例1：自己指定卷积核参数

![自己指定卷积核参数](C:\Users\acer-pc\Desktop\办公\自己写的资料\picture\pytorch常用函数\自己指定卷积核参数.png)

示例2：

```
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
>>> output = m(input)
```



### 1.8 torch.nn.Conv1d

- class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

- 一维卷积层，输入的尺度是(N, C_in,L)，输出尺度（ N,C_out,L_out）的计算方式：
- in_channels(`int`) – 输入信号的通道
- out_channels(`int`) – 卷积产生的通道
- kerner_size(`int` or `tuple`) - 卷积核的尺寸
- stride(`int` or `tuple`, `optional`) - 卷积步长
- padding (`int` or `tuple`, `optional`)- 输入的每一条边补充0的层数
- bias(`bool`, `optional`) - 如果`bias=True`，添加偏置
- 输入: (N,C_in,L_in)
  输出: (N,C_out,L_out)
- weight(`tensor`) - 卷积的权重，大小是(`out_channels`, `in_channels`, `kernel_size`)
  bias(`tensor`) - 卷积的偏置系数，大小是（`out_channel`）

示例

```
>>> m = nn.Conv1d(16, 33, 3, stride=2)
>>> input = autograd.Variable(torch.randn(20, 16, 50))
>>> output = m(input)
```



### 1.9 torch.nn.MaxPool2d 

- class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
- 对于输入信号的输入通道，提供2维最大池化（`max pooling`）操作
- 如果输入的大小是(N,C,H,W)，那么输出的大小是(N,C,H_out,W_out)
- kernel_size(`int` or `tuple`) - max pooling的窗口大小
- stride(`int` or `tuple`, `optional`) - max pooling的窗口移动的步长。默认值是`kernel_size`
- padding(`int` or `tuple`, `optional`) - 输入的每一条边补充0的层数



### 1.10 torch.cuda.is_available()

- 返回值为布尔类型，确认是否可以用gpu加速。



### 1.11 torch.device()

![torch.device](C:\Users\acer-pc\Desktop\办公\自己写的资料\picture\pytorch常用函数\torch.device.png)

![train process](C:\Users\acer-pc\Desktop\办公\自己写的资料\picture\pytorch常用函数\train process.png)

![test process](C:\Users\acer-pc\Desktop\办公\自己写的资料\picture\pytorch常用函数\test process.png)

- torch.device内的“cuda:0”意思是加入存在gpu的情况，指定第0块显卡，一个机器可能有多个显卡跑多个任务。
- model.to(device)意思是把整个模型的可训练参数和缓存转换成cuda tensor，把cpu建立的权重迁移到gpu上。
- inputs.to(device),target.to(device)意思是把需要计算的tensor也迁移到gpu上，主要是输入和输出，比注意model的可训练参数和需要计算的tensor必须要在同一块显卡上。



### 1.12 torch.optim.lr_scheduler()

`torch.optim.lr_scheduler`模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。

参考自[csdn](https://blog.csdn.net/qyhaill/article/details/103043637)



### 1.13 torch.nn.MarginRankingLoss()

- 功能： 计算两个向量之间的相似度，当两个向量之间的距离大于margin，则loss为正，小于margin，loss为0。
- transE模型用的评分函数就是这个

参考自[csdn](https://blog.csdn.net/qq_38813456/article/details/110083290)



## 2. 较为常用

### 2.1 register_parameter(name, param)

向`module`添加 `parameter`

`parameter`可以通过注册时候的`name`获取。

- 实现的功能是将一个不可训练的tensor变为module内的可训练类型parameter。

> https://blog.csdn.net/xinjieyuan/article/details/106951116?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162797769716780271520540%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162797769716780271520540&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-106951116.first_rank_v2_pc_rank_v29&utm_term=register_parameter%28%29&spm=1018.2226.3001.4187



### 2.2 torch.zeros()

```
torch.zeros(size, out=None)
```

**参数**：
**size**：定义输出张量形状的整数序列
**out (Tensor, optional)**：输出张量



### 2.3 torch.sort()

```
torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
```

对输入张量`input`沿着指定维按升序排序。如果不给定`dim`，则默认为输入的最后一维。如果指定参数`descending`为`True`，则按降序排序

返回元组 (sorted_tensor, sorted_indices) ， `sorted_indices` 为原始输入中的下标。

参数:

- input (Tensor) – 要对比的张量
- dim (int, optional) – 沿着此维排序
- descending (bool, optional) – 布尔值，控制升降排序
- out (tuple, optional) – 输出张量。必须为`ByteTensor`或者与第一个参数`tensor`相同类型。

> https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchsort

### 2.4 torch.FloatTensor()

- 类型转换, 将list ,numpy转化为tensor。 以list -> tensor为例：

```
print(torch.FloatTensor([1,2]))
# 输出: tensor([1., 2.])
```

- 根据输入大小创建一个空tensor

![img](https://img-blog.csdnimg.cn/20190219235621776.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM4NjY2,size_16,color_FFFFFF,t_70)



### 2.5 tensor.squeeze()和unsqueeze()

- **torch.squeeze()** 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度
- **torch.unsqueeze()**这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度

1. 首先得到一个维度为（1，2，3）的tensor（张量）

![img](https://img-blog.csdn.net/20180812160833709?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZseXNreV9qYXk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

由图中可以看出c的维度为（1，2，3）

2.下面使用squeeze()函数将第一维去掉

![img](https://img-blog.csdn.net/20180812161010282?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZseXNreV9qYXk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

可见，维度已经变为（2，3）

3.另外

![img](https://img-blog.csdn.net/20180812161246184?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZseXNreV9qYXk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

可以看出维度并没有变化，仍然为（1，2，3），这是因为只有维度为1时才会去掉。


> https://blog.csdn.net/flysky_jay/article/details/81607289



### 2.6 torch.mul()、torch.mm()、torch.matmul()

1. torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵；
2. torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。
3. torch.matmul()也是一种类似于矩阵相乘操作的tensor联乘操作。但是它可以利用python 中的广播机制，处理一些维度不同的tensor结构进行相乘操作。

> https://www.jianshu.com/p/e277f7fc67b3



### 2.7 torch.norm()

```
torch.norm(input,p="fro",dim=None)
```

- torch.norm是对输入的Tensor求范数
- input (Tensor) – 输入张量
- p (float,optional) – 范数计算中的幂指数值
- dim (int) – 缩减的维度

参考自[csdn](https://blog.csdn.net/goodxin_ie/article/details/84657975)



### 2.8 torch.repeat()

torch.repeat用法与np.tile()类似

```
import torch
x = torch.tensor([1, 2, 3])
print(x.repeat(4, 1))
print("###################################")
print(x.repeat(4, 2))

```

运行结果为

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216204552555.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQyMjEyNjY=,size_16,color_FFFFFF,t_70)

参考自[csdn](https://blog.csdn.net/appleml/article/details/103569615)



### 2.9 torch.expand_as()

据说用法与expand()类似，把一个tensor变成和函数括号内一样形状的tensor

```
>>> x = torch.tensor([[1], [2], [3]])
>>> x.size()
torch.Size([3, 1])
>>> x.expand(3, 4)
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
>>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
tensor([[ 1,  1,  1,  1],
        [ 2,  2,  2,  2],
        [ 3,  3,  3,  3]])
```

参考自[csdn](https://blog.csdn.net/lcqin111/article/details/89765081)



### 2.10 torch.unique()

```
import torch
 
x = torch.tensor([4,0,1,2,1,2,3])#生成一个tensor,作为实验输入
print(x)
 
out = torch.unique(x) #所有参数都设置为默认的
print(out)#将处理结果打印出来
#结果如下：
#tensor([0, 1, 2, 3, 4])   #将x中的不重复元素挑了出来，并且默认为生序排列
 
out = torch.unique(x,sorted=False)#将默认的生序排列改为False
print(out)
#输出结果如下：
#tensor([3, 2, 1, 0, 4])  #将x中的独立元素找了出来，就按照原始顺序输出
 
out = torch.unique(x,return_inverse=True)#将原始数据中的每个元素在新生成的独立元素张量中的索引输出
print(out)
#输出结果如下：
#(tensor([0, 1, 2, 3, 4]), tensor([4, 0, 1, 2, 1, 2, 3]))  #第一个张量是排序后输出的独立张量，第二个结果对应着原始数据中的每个元素在新的独立无重复张量中的索引，比如x[0]=4,在新的张量中的索引为4, x[1]=0,在新的张量中的索引为0，x[6]=3,在新的张量中的索引为3
 
out = torch.unique(x,return_counts=True) #返回每个独立元素的个数
print(out)
#输出结果如下
#(tensor([0, 1, 2, 3, 4]), tensor([1, 2, 2, 1, 1]))  #0这个元素在原始数据中的数量为1,1这个元素在原始数据中的数量为2
```

参考自[csdn](https://blog.csdn.net/t20134297/article/details/108235355)



### 2.11 torch.spares_coo_tensor()

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201111193328567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZsdWlkX3JheQ==,size_16,color_FFFFFF,t_70#pic_center)

```
torch.spares_coo_tensor(indices, values, siez=None,*, dtype=None, requires_grad=False)->Tensor

```

- indices：此参数是指定非零元素所在的位置，也就是行和列，所以此参数应该是一个二维的数组，当然它可以是很多格式（ist, tuple, NumPy ndarray, scalar, and other types. ）第一维指定了所有非零数所在的行数，第二维指定了所有非零元素所在的列数。例如indices=[[1, 4, 6], [3, 6, 7]]表示我们稀疏矩阵中(1, 3),(4, 6), (6, 7)几个位置是非零的数所在的位置。
- values：此参数指定了非零元素的值，所以此矩阵长度应该和上面的indices一样长也可以是很多格式（list, tuple, NumPy ndarray, scalar, and other types.）。例如``values=[1, 4, 5]表示上面的三个位置非零数分别为1, 4, 5。
- size：指定了稀疏矩阵的大小，例如size=[10, 10]表示矩阵大小为10 × 10 10\times 1010×10，此大小最小应该足以覆盖上面非零元素所在的位置，如果不给定此值，那么默认是生成足以覆盖所有非零值的最小矩阵大小。
- `dtype`：指定返回tensor中数据的类型，如果不指定，那么采取values中数据的类型。
- `device`：指定创建的tensor在cpu还是cuda上。
- `requires_grad`：指定创建的tensor需不需要梯度信息，默认为`False`

示例代码

```
import torch

indices = torch.tensor([[4, 2, 1], [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(indices=indices, values=values, size=[5, 5])
x
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201111195443623.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZsdWlkX3JheQ==,size_16,color_FFFFFF,t_70#pic_center)

参考自[csdn](https://blog.csdn.net/Fluid_ray/article/details/109629482)



### 2.12 tensor.T和tensor.t()

都表示为对tensor的转置。

-  **.t()** 是 **.transpose**函数的简写版本，但只能对2维以下的tensor进行**转置**。
-  **.T** 是 **.permute** 函数的简化版本，不仅可以操作2维tensor，甚至可以对n维tensor进行转置。当然当维数n=2时，**.t()** 与 **.T** 效果是一样的。

参考自[csdn](https://blog.csdn.net/lollows/article/details/105017813)

