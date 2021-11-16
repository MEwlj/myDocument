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



### 1.14 nn.LSTMCell

```
import torch
from torch import nn

# 一层的LSTM计算单元,输入的feature_len=100,隐藏单元和记忆单元hidden_len=20
cell = nn.LSTMCell(input_size=100, hidden_size=20)

# 初始化隐藏单元h和记忆单元C,取batch=3
h = torch.zeros(3, 20)
C = torch.zeros(3, 20)

# 这里是seq_len=10个时刻的输入,每个时刻shape都是[batch,feature_len]
xs = [torch.randn(3, 100) for _ in range(10)]

# 对每个时刻,传入输入x_t和上个时刻的h_{t-1}和C_{t-1}
for xt in xs:
    h, C = cell(xt, (h, C))

print(h.shape)  # torch.Size([3, 20])
print(C.shape)  # torch.Size([3, 20])

```

[csdn](https://blog.csdn.net/SHU15121856/article/details/104448734)



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

4. torch.bmm

```
torch.bmm(input, mat2, out=None) → Tensor
```

 torch.bmm()是tensor中的一个相乘操作，类似于矩阵中的A*B。

参数：

input，mat2：两个要进行相乘的tensor结构，两者必须是3D维度的

output：输出结果

并且相乘的两个矩阵，要满足一定的维度要求：input（p,m,n) * mat2(p,n,a) ->output(p,m,a)。这个要求，可以类比于矩阵相乘。前一个矩阵的列等于后面矩阵的行才可以相乘。
————————————————
版权声明：本文为CSDN博主「Foneone」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/foneone/article/details/103876519



补充：

1. 矩阵相乘有`torch.mm(a, b)`和`torch.matmul(a, b)`两个函数。

   前一个是针对二维矩阵，后一个是高维。当`torch.mm(a, b)`用于大于二维时将报错。

2. matmul对于维数相同的张量

   A.shape =（b,m,n)；B.shape = (b,n,k)
   numpy.matmul(A,B) 结果shape为(b,m,k)

   要求第一维度相同，后两个维度能满足矩阵相乘条件。

3. matmul对于维数不同的张量

   比如 A.shape =（**m,n**)； B.shape = (b,**n,k**)； C.shape=(**k,l**)

   numpy.matmul(A,B) 结果shape为(b,m,k)

   numpy.matmul(B,C) 结果shape为(b,n,l)

   2D张量要和3D张量的后两个维度满足矩阵相乘条件。

摘抄自[csdn](https://blog.csdn.net/qq_34243930/article/details/106889639)

有一个说法，对于高纬数组，乘法只作用在最后两维（看最后两维是否可以进行乘法），这一点在torch.mutmul验证过了，是正确的，比如说

```
torch.matmul(att,tail).squeeze(2)
#att:tensor:(1024,5,1,50),tail:tensor:(1024,5,50,100)
#结果为tensor：（1024,5,100）
```

[知乎](https://zhuanlan.zhihu.com/p/203085276)



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

这个函数如函数名一样，是复制函数，参数表示把这个tensor复制成多少个，参数以1,2,3位来解释：

假设a是一个tensor，那么把a看作最小单元：

a.repeat(2)表示在复制1行2列a;

a.repeat(3, 2)表示复制3行2列个a；

a.repeat(3, 2, 1)表示复制3个2行1列个a。



作者：不太聪明的亚子
链接：https://www.jianshu.com/p/206ef7cba355
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



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



### 2.13 F.normalize

```
torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
```

1. 输入为一维tensor

```
a = torch.Tensor([1,2,3])

torch.nn.functional.normalize(a, dim=0)

tensor([0.2673, 0.5345, 0.8018])

```

可以看到每一个数字都除以了这个Tensor的范数：$\sqrt{1^2+2^2+3^2}=3.7416$

2. 输入为二维tensor

```
b = torch.Tensor([[1,2,3], [4,5,6]])

torch.nn.functional.normalize(b, dim=0)

tensor([[0.2425, 0.3714, 0.4472],
        [0.9701, 0.9285, 0.8944]])

```

因为dim=0，所以是对列操作。每个数除以所在列的范数。第一列的范数为$\sqrt{1^2+4^2}=4.1231$



内容节选自[csdn](https://blog.csdn.net/ECNU_LZJ/article/details/103653133?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163248324716780255227204%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163248324716780255227204&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-103653133.first_rank_v2_pc_rank_v29&utm_term=torch.nn.functional.normalize&spm=1018.2226.3001.4187)



### 2.14 torch.view

1. torch.view()类似于numpy中的resize(),返回的tensor和传入的tensor共享内存，意思就是修改其中一个，数据都会变.
2. torch.view()的形状可以输入-1，意思是计算机自动帮我们计算对应的数字

```
import torch
a = torch.arange(0,20)		#此时a的shape是(1,20)
a.view(4,5).shape		#输出为(4,5)
a.view(-1,5).shape		#输出为(4,5)
a.view(4,-1).shape		#输出为(4,5)
```

内容摘抄自[博客园](https://www.cnblogs.com/MartinLwx/p/10543604.html)



### 2.15 torch.sum

torch.sum()对输入的tensor数据的某一维度求和，一共两种用法

1. ```
   １．torch.sum(input, dtype=None)
   ２．torch.sum(input, list: dim, bool: keepdim=False, dtype=None) → Tensor
   　
   input:输入一个tensor
   dim:要求和的维度，可以是一个列表
   keepdim:求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True
   #If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. 
   ```

2. ```
   a = torch.ones((2, 3))
   print(a):
   tensor([[1, 1, 1],
    		[1, 1, 1]])
   
   a1 =  torch.sum(a)
   a2 =  torch.sum(a, dim=0)
   a3 =  torch.sum(a, dim=1)
   
   print(a)
   print(a1)
   print(a2)
   ```

输出结果为

```
tensor(6.)
tensor([2., 2., 2.])
tensor([3., 3.])
```

选自[csdn](https://blog.csdn.net/qq_39463274/article/details/105145029)



### 2.16 torch.narrow

1. 从 input 张量中返回一个范围限制后的 张量，范围限制条件为：沿维度dim 从 start 到start+length 的范围区间(闭区间)，类似于数组切片用法，返回的张量与 input 张量共享相同储存基础
   参数

2. input(Tensor) ，需处理的张量；
3. dim(int)，沿着限制的轴；
4. start(int) ，张量起始点；
   length(int) ，缩窄长度;

```
rand_float = torch.randn((5,3))# 随机生成 5*3数据
rand_float
>>>
tensor([[-0.4972, -0.1363, -1.8918],
        [ 1.2994, -1.0091,  0.1862],
        [ 0.5525,  1.3073,  1.3741],
        [-1.7242, -0.3593, -0.7546],
        [-0.3328,  0.3333,  0.0096]])
        
rand_float.narrow(0,1,2)# 沿第一维度开始，第一行为开始，长度为2
>>>
tensor([[ 1.2994, -1.0091,  0.1862],
        [ 0.5525,  1.3073,  1.3741]])

```

选自[csdn](https://blog.csdn.net/weixin_42512684/article/details/110789511)，内容参考自[博客园](https://www.cnblogs.com/qinduanyinghua/p/11862641.html)



### 2.17 F.pad

**torch.nn.functional.pad(input, pad, mode='constant', value=0)**

- pad
  扩充维度，用于预先定义出某维度上的扩充参数
- mode
  扩充方法，’constant‘, ‘reflect’ or ‘replicate’三种模式，分别表示常量，反射，复制
- value
  扩充时指定补充值，但是value只在mode='constant’有效，即使用value填充在扩充出的新维度位置，而在’reflect’和’replicate’模式下，value不可赋值

**当pad只有两个参数时，仅改变最后一个维度**

![img](https://pic3.zhimg.com/80/v2-ff4e9a64d32fb17d7a770268582a970a_720w.jpg)

摘抄自[知乎](https://zhuanlan.zhihu.com/p/358599463)



### 2.18 torch.randn 和 torch.rand

```
torch.randn(*sizes, out=None) → Tensor

```

返回一个包含了从**标准正态分布**中抽取的一组随机数的张量

size：张量的形状，

out:结果张量。（目前还没有看到使用这个参数的例子）

```
torch.rand(*sizes, out=None) → Tensor
```

但是它是**[0,1)之间的均匀分布**

[博客](https://www.cnblogs.com/jiading/p/11944458.html)



### 2.19 tensor.masked_fill_(mask,value)

masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充

```
>>> t = torch.randn(3,2)
>>> t
tensor([[-0.9180, -0.4654],
        [ 0.9866, -1.3063],
        [ 1.8359,  1.1607]])
>>> m = torch.randint(0,2,(3,2))
>>> m
tensor([[0, 1],
        [1, 1],
        [1, 0]])
>>> m == 0
tensor([[ True, False],
        [False, False],
        [False,  True]])
>>> t.masked_fill(m == 0, -1e9)
tensor([[-1.0000e+09, -4.6544e-01],
        [ 9.8660e-01, -1.3063e+00],
        [ 1.8359e+00, -1.0000e+09]])
```

csdn的[anshiquanshu](https://blog.csdn.net/anshiquanshu/article/details/111376283)



### 2.19 torch.einsum

三条基本规则

首先看下 einsum 实现矩阵乘法的例子：

```
a = torch.rand(2,3)
b = torch.rand(3,4)
c = torch.einsum("ik,kj->ij", [a, b])
# 等价操作 torch.mm(a, b)
```

其中需要重点关注的是 einsum 的第一个参数 "ik,kj->ij"，该字符串（下文以 equation 表示）表示了输入和输出张量的维度。equation 中的箭头左边表示输入张量，以逗号分割每个输入张量，箭头右边则表示输出张量。表示维度的字符只能是26个英文字母 'a' - 'z'。

而 einsum 的第二个参数表示实际的输入张量列表，其数量要与 equation 中的输入数量对应。同时对应每个张量的 子 equation 的字符个数要与张量的真实维度对应，比如 "ik,kj->ij" 表示输入和输出张量都是两维的。

equation  中的字符也可以理解为索引，就是输出张量的某个位置的值，是怎么从输入张量中得到的，比如上面矩阵乘法的输出 c 的某个点 c[i, j] 的值是通过 a[i, k] 和 b[i, k] 沿着 k 这个维度做内积得到的。

接着介绍两个基本概念，自由索引（ *Free indices* ）和求和索引（ *Summation indices* ）：

- 自由索引，出现在箭头右边的索引，比如上面的例子就是 i 和 j；
- 求和索引，只出现在箭头左边的索引，表示中间计算结果需要这个维度上求和之后才能得到输出，比如上面的例子就是 k；

接着是介绍三条基本规则：

- 规则一，equation 箭头左边，在 **不同** 输入之间重复出现的索引表示，把输入张量沿着该维度做乘法操作，比如还是以上面矩阵乘法为例， "ik,kj->ij"，k 在输入中重复出现，所以就是把 a 和 b 沿着 k 这个维度作相乘操作；
- 规则二，只出现在 equation 箭头左边的索引，表示中间计算结果需要在这个维度上求和，也就是上面提到的求和索引；
- 规则三，equation 箭头右边的索引顺序可以是任意的，比如上面的 "ik,kj->ij" 如果写成 "ik,kj->ji"，那么就是返回输出结果的转置，用户只需要定义好索引的顺序，转置操作会在 einsum 内部完成。

> https://www.freeaihub.com/post/105614.html



### 2.20 torch.split()

```
torch.split(tensor, split_size_or_sections, dim=0)
```

  torch.split()作用将tensor分成块结构。

参数：

tesnor：input，待分输入
split_size_or_sections：需要切分的大小(int or list )
dim：切分维度
output：切分后块结构 <class 'tuple'>
当split_size_or_sections为int时，tenor结构和split_size_or_sections，正好匹配，那么ouput就是大小相同的块结构。如果按照split_size_or_sections结构，tensor不够了，那么就把剩下的那部分做一个块处理。
当split_size_or_sections 为list时，那么tensor结构会一共切分成len(list)这么多的小块，每个小块中的大小按照list中的大小决定，其中list中的数字总和应等于该维度的大小，否则会报错（注意这里与split_size_or_sections为int时的情况不同）。
————————————————
版权声明：本文为CSDN博主「skycrygg」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_42518956/article/details/103882579

- **split_size_or_sections为\**int\**型时**

```
import torch
 
x = torch.rand(4,8,6)
y = torch.split(x,2,dim=0) #按照4这个维度去分，每大块包含2个小块
for i in y :
    print(i.size())
 
output:
torch.Size([2, 8, 6])
torch.Size([2, 8, 6])
 
y = torch.split(x,3,dim=0)#按照4这个维度去分，每大块包含3个小块
for i in y:
    print(i.size())
 
output:
torch.Size([3, 8, 6])
torch.Size([1, 8, 6])
```

-  **split_size_or_sections为\**list\**型时。**

```
import torch
 
x = torch.rand(4,8,6)
y = torch.split(x,[2,3,3],dim=1)
for i in y:
    print(i.size())
 
output:
torch.Size([4, 2, 6])
torch.Size([4, 3, 6])
torch.Size([4, 3, 6])
 
 
y = torch.split(x,[2,1,3],dim=1) #2+1+3 等于6 != 8 ,报错
for i in y:
    print(i.size())
 
output:
split_with_sizes expects split_sizes to sum exactly to 8 (input tensor's size at dimension 1), but got split_sizes=[2, 1, 3]
```

ps:与torch.chunk不一样，chunk的第二个参数均匀分割的份数，如果该tensor在你要进行分割的维度上的size不能被chunks整除，则最后一份会略小（也可能为空）

参考自[博客园](https://www.cnblogs.com/moon3/p/12685911.html)，注意，博客园关于torch.split说错了一点，当第二个参数为int类型时，并不是均匀分割的份数，而是按该int数值进行分割。

