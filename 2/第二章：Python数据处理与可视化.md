---
marp: true
theme: gaia
author: Shiyan Pan
size: 4:3
footer: '2023-09-16'
header: '第二章：Python数据处理与可视化'
paginate: true
style: |
  section a {
      font-size: 100px;
  }
---

# 第二章：Python数据处理与可视化
## 2.1 △ NumPy简介
## 2.2 △ Pandas简介
## 2.3 数据获取
## 2.4 数据清洗和预处理
## 2.5 △ 数据可视化（Matplotlib）

---

## 2.1 NumPy简介
- 矩阵、张量和数组

<font size=5>

  - 矩阵（Matrix）：
    - 定义： 矩阵是一个由数字排列成的矩形网格，其中包含了行和列。通常表示为一个二维数组，其中每个元素都有行和列的坐标。
    - 用途： 矩阵在线性代数中广泛用于矩阵乘法、线性方程组求解、特征值计算等。    
    - 示例： 以下是一个2x3的矩阵的示例：
```   
      [1  2  3]
      [4  5  6]
```       

</font>   

---

## 2.1 NumPy简介
- 矩阵、张量和数组

<font size=5>

  - 张量（Tensor）：
    - 定义： 张量是一个多维数组，通常包含了三个或更多维度。在深度学习和神经网络中，张量是表示多层数据的常用数据结构。
    - 用途： 张量在深度学习中用于表示神经网络的输入、输出和权重，以及在图像处理、自然语言处理等领域中用于处理多维数据。    
    - 示例： 以下是一个3D张量的示例，表示RGB图像

```  
  [[[255  0  0]
    [0    255  0]
    [0    0    255]]
    
  [[128  128  0]
    [0    128  128]
    [128  0    128]]]

```  
</font>   

---

## 2.1 NumPy简介
- 矩阵、张量和数组

<font size=5>

  - 数组（Array）：
    - 定义： 数组是一种通用的多维数据结构，可以包含任意数量的维度。在Python中，NumPy库提供了强大的数组功能，被广泛用于科学计算和数据分析。
    - 用途： 数组在科学计算、数据分析、机器学习、统计分析等领域中广泛用于存储和处理数据。
    - 示例： 以下是一个2D NumPy数组的示例：

```
[[1  2  3]
 [4  5  6]]
```

</font>   

---



## 2.1 NumPy简介
- 张量和数组关系

<font size=5>

![width:800](2.1.png)

</font>   

---


## 2.1 NumPy简介
- 从标量到向量到矩阵到张量

<font size=5>


![width:700](2.3.png)

</font>   

---





## 2.1 NumPy简介
- 矩阵和张量Python表示

<font size=5>

![width:800](2.2.png)

</font>   

---







## 2.1 NumPy简介
- 矩阵、张量和数组

<font size=4>

| 特征          | 矩阵                 | 张量                      | 数组                   |
| ------------- | -------------------- | ------------------------- | ---------------------- |
| 数据结构      | 二维数据结构         | 多维数据结构               | 多维数据结构            |
| 维度数量      | 2                    | 通常大于2                 | 通常大于2              |
| 元素类型      | 通常是数值           | 通常是数值                | 通常是数值             |
| 常见用途      | 线性代数运算         | 深度学习和神经网络         | 科学计算、数据分析     |
| 维度表示      | 通常用行和列表示     | `通常用轴（axis）表示`      | `通常用轴（axis）表示`   |
| 示例          | 2x2、3x3 矩阵       | 3D、4D、5D 张量           | 1D、2D、3D 数组        |

</font>  

---


## 2.1 NumPy简介
- NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
- NumPy 的前身 Numeric 最早是由 Jim Hugunin 与其它协作者共同开发，2005 年，Travis Oliphant 在 Numeric 中结合了另一个同性质的程序库 Numarray 的特色，并加入了其它扩展而开发了 NumPy。NumPy 为开放源代码并且由许多协作者共同维护开发。

---

## 2.1 NumPy简介
- NumPy 是一个运行速度非常快的数学库，主要用于数组计算，包含：
  - 一个强大的N维数组对象 ndarray
  - 广播功能函数
  - 整合 C/C++/Fortran 代码的工具
  - 线性代数、傅里叶变换、随机数生成等功能

---

## 2.1 NumPy简介
- NumPy 应用
  - NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用， 这种组合广泛用于替代 MatLab，是一个强大的科学计算环境，有助于我们通过 Python 学习数据科学或者机器学习。

  - SciPy 是一个开源的 Python 算法库和数学工具包。

  - SciPy 包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。

  - Matplotlib 是 Python 编程语言及其数值数学扩展包 NumPy 的可视化操作界面。它为利用通用的图形用户界面工具包，如 Tkinter, wxPython, Qt 或 GTK+ 向应用程序嵌入式绘图提供了应用程序接口（API）。

---  

## 2.1 NumPy简介
- Anaconda虚拟环境下使用pip安装numpy

  pip install numpy

- 安装验证

```python
  from numpy import *  # 导入 numpy 库。

  eye(4) # 生成对角矩阵。

  array([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
```

--- 


## 2.1 NumPy简介
- Ndarray 对象
  - NumPy 最重要的一个特点是其 N 维数组对象 ndarray，它是一系列同类型数据的集合，以 0 下标为开始进行集合中元素的索引。
  - ndarray 对象是用于存放同类型元素的多维数组。
  - ndarray 中的每个元素在内存中都有相同存储大小的区域。

--- 

## 2.1 NumPy简介
- Ndarray 对象
  - ndarray 内部由以下内容组成：

<font size=4>

| 描述                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| 指针（Pointer）          | 一个指向数据（内存或内存映射文件中的一块数据）的指针。     |
| 数据类型或 dtype        | 描述在数组中的固定大小值的格子。                             |
| 形状（Shape）            | 一个表示数组形状的元组，表示各维度大小的元组。               |
| 跨度元组（Stride Tuple） | 一个跨度元组（stride），其中的整数指的是为了前进到当前维度下一个元素需要"跨过"的字节数。 |

</font>

--- 

## 2.1 NumPy简介
- Ndarray 对象
ndarray 的内部结构:

![width:800](2/2.1.jpg)

<font size=4>
跨度可以是负数，这样会使数组在内存中后向移动，切片中 obj[::-1] 或 obj[:,::-1] 就是如此。
</font>

--- 

## 2.1 NumPy简介
- Ndarray 对象
创建一个 ndarray
```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```
<font size=4>

| 名称    | 描述                                           |
| ------- | ---------------------------------------------- |
| object  | 数组或嵌套的数列                               |
| dtype   | 数组元素的数据类型，可选                       |
| copy    | 对象是否需要复制，可选                         |
| order   | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
| subok   | 默认返回一个与基类类型一致的数组               |
| ndmin   | 指定生成数组的最小维度                         |

</font>

--- 

## 2.1 NumPy简介
```python
import numpy as np 
a = np.array([1,2,3])  
print (a)
```
<font size=5>

输出结果：
```
[1 2 3]
```

| 轴 (Axis) | 大小 (Size) | 内容   |
| --------- | ----------- | ------ |
| 0         | 3           | 1, 2, 3 |

</font>

--- 


## 2.1 NumPy简介
```python
import numpy as np 
a = np.array([[1,  2],  [3,  4]])  
print (a)
```
<font size=5>

输出结果：
```
[[1  2] 
 [3  4]]
```

| 轴 (Axis) | 大小 (Size)   | 内容       |
| --------- | ------------- | ---------- |
| 0         | 2             | [1 2] 和 [3 4]   |
| 1         | 2             | 1 和 2，3 和 4   |

</font>

--- 


## 2.1 NumPy简介
```python
# 最小维度  
import numpy as np 
a = np.array([1, 2, 3, 4, 5], ndmin =  2)  
print (a)
```
<font size=5>

输出结果：
```
[[1 2 3 4 5]]
```

| 轴 (Axis) | 大小 (Size) | 内容         |
| --------- | ----------- | ------------ |
| 0         | 1           | [1 2 3 4 5]  |
| 1         | 5           | 1, 2, 3, 4, 5 |

</font>

--- 


## 2.1 NumPy简介
```python
# dtype 参数  
import numpy as np 
a = np.array([1,  2,  3], dtype = complex)  
print (a)
```
<font size=5>

输出结果：
```
[1.+0.j 2.+0.j 3.+0.j]
```
| 轴 (Axis) | 大小 (Size) | 内容              |
| --------- | ----------- | ----------------- |
| 0         | 3           | 1.+0.j, 2.+0.j, 3.+0.j |

</font>

--- 


## 2.1 NumPy简介
```python
三维数组:
 [[[1 2] 7] [3 4]]
```

| 轴 (Axis) | 大小 (Size) | 内容       |
| --------- | ----------- | ---------- |
| 0         | 2           | [[1 2] 7] [3 4] |
| 1         | 4           | [1 2] 和 7 以及 3 和 4 |
| 2         | 2           | 1 和 2 |


--- 

## 2.1 NumPy简介
```python
三维数组:
[[[1  2]   7] 
 [3  [8  3]]]
```

| 轴 (Axis) | 大小 (Size) | 内容       |
| --------- | ----------- | ---------- |
| 0         | 2           | [[1 2] 7] 和 [3 [8  3]] |
| 1         | 4           | [1 2] 和 7 以及 3 和 [8  3] |
| 2         | 4           | 1 和 2 以及 8 和 3|


--- 

## 2.1 NumPy简介
```python
四维数组:
[[[1  [9  2]]   7] 
 [3  [[4  4]  3]]]
```

<font size=5>

| 轴 (Axis) | 大小 (Size) | 内容       |
| --------- | ----------- | ---------- |
| 0         | 2           | [[1 [9  2]] 7] 和 [3 [[4  4]  3]] |
| 1         | 4           | [1 [9  2]] 和 7 以及 3 和 [[4  4]  3] |
| 2         | 4           | 1 和 [9  2] 以及 [4  4] 和 3|
| 3         | 4           | 9 和 2 以及 4 和 4|

</font>

--- 

## 2.1 NumPy简介

- NumPy 数据类型
numpy 支持的数据类型比 Python 内置的类型要多很多，基本上可以和 C 语言的数据类型对应上，其中部分类型对应为 Python 内置的类型。下表列举了常用 NumPy 基本类型。

bool_, int_, intc, intp, int8, int16, int32, int64,...

---

## 2.1 NumPy简介

- NumPy 数据类型
  - 数据类型对象（numpy.dtype 类的实例）用来描述与数组对应的内存区域是如何使用。

<font size=5>

| 属性                   | 描述                               |
| -----------------------| ---------------------------------- |
| 数据的类型             | 整数、浮点数或 Python 对象等       |
| 数据的大小             | 数据类型占用多少字节存储          |
| 数据的字节顺序         | 小端法（Little Endian）或大端法（Big Endian） |
| 结构化类型字段         | 字段名称、每个字段的数据类型、每个字段所占内存块的部分 |
| 子数组的形状和数据类型 | 子数组的形状（shape）和数据类型（dtype） |
 
</font>

---

## 2.1 NumPy简介
- NumPy 数据类型

```python
import numpy as np
# 使用标量类型
dt = np.dtype(np.int32)
print(dt)

输出结果：

int32
```

---

## 2.1 NumPy简介
- NumPy 数据类型

```python
import numpy as np
# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
dt = np.dtype('i4')
print(dt)
输出结果为：

int32
```

---

## 2.1 NumPy简介
- NumPy 数据类型

```python
import numpy as np
# 字节顺序标注
dt = np.dtype('<i4')
print(dt)
输出结果为：

int32
```

---

## 2.1 NumPy简介
- NumPy 数据类型

```python
# 首先创建结构化数据类型
import numpy as np
dt = np.dtype([('age',np.int8)]) 
print(dt)
输出结果为：

[('age', 'i1')]
```

---



## 2.1 NumPy简介
- NumPy 数组属性

<font size=5>

  - NumPy 数组的维数称为秩（rank），秩就是轴的数量，即数组的维度，一维数组的秩为 1，二维数组的秩为 2，以此类推。
  - 在 NumPy中，每一个线性的数组称为是一个轴（axis），也就是维度（dimensions）。比如说，二维数组相当于是两个一维数组，其中第一个一维数组中每个元素又是一个一维数组。所以一维数组就是 NumPy 中的轴（axis），第一个轴相当于是底层数组，第二个轴是底层数组里的数组。而轴的数量——秩，就是数组的维数。
  - 很多时候可以声明 axis。axis=0，表示沿着第 0 轴进行操作，即对每一列进行操作；axis=1，表示沿着第1轴进行操作，即对每一行进行操作。

</font>

---



## 2.1 NumPy简介
- NumPy 数组属性
<font size=5>
  - NumPy 的数组中比较重要 ndarray 对象属性有：

| 属性               | 说明                                               |
| -------------------| -------------------------------------------------- |
| `ndarray.ndim`      | 秩，即轴的数量或维度的数量                        |
| `ndarray.shape`      | 数组的维度，对于矩阵，n 行 m 列                   |
| `ndarray.size`       | 数组元素的总个数，相当于 .shape 中 n*m 的值       |
| ndarray.dtype      | ndarray 对象的元素类型                            |
| ndarray.itemsize   | ndarray 对象中每个元素的大小，以字节为单位       |
| ndarray.flags      | ndarray 对象的内存信息                            |
| ndarray.real       | ndarray 元素的实部                                |
| ndarray.imag       | ndarray 元素的虚部                                |
| ndarray.data       | 包含实际数组元素的缓冲区，通常不需要使用         |

</font>

---

## 2.1 NumPy简介
- ndarray.ndim

<font size=5>

  - ndarray.ndim 用于返回数组的维数，等于秩。

```python
import numpy as np 
 
a = np.arange(24)  
print (a.ndim)             # a 现只有一个维度
# 现在调整其大小
b = a.reshape(2,4,3)  # b 现在拥有三个维度
print (b.ndim)

输出结果为：

1
3
```

</font>

---

## 2.1 NumPy简介
- ndarray.shape

<font size=5>

  - ndarray.shape 表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 ndim 属性(秩)。比如，一个二维数组，其维度表示"行数"和"列数"。
  - ndarray.shape 也可以用于调整数组大小。

```python
import numpy as np  
 
a = np.array([[1,2,3],[4,5,6]])  
print (a.shape)

输出结果为：

(2, 3)
```

</font>

---


## 2.1 NumPy简介
- ndarray.shape

<font size=5>

  - 调整数组大小。

```python
import numpy as np 
 
a = np.array([[1,2,3],[4,5,6]]) 
a.shape =  (3,2)  
print (a)

输出结果为：

[[1 2]
 [3 4]
 [5 6]]
```

</font>

---

## 2.1 NumPy简介
- ndarray.shape

<font size=5>

  - reshape 函数调整数组大小。

```python
import numpy as np 
 
a = np.array([[1,2,3],[4,5,6]]) 
b = a.reshape(3,2)  
print (b)

输出结果为：

[[1 2]
 [3 4]
 [5 6]]
```

</font>

---


## 2.1 NumPy简介
- NumPy 创建数组
  - numpy.empty 创建一个指定形状（shape）、数据类型（dtype）且`未初始化`的数组：
```python
  numpy.empty(shape, dtype = float, order = 'C')
```

<font size=5>

| 参数   | 描述                                        |
| -------| ------------------------------------------- |
| shape  | 数组形状                                    |
| dtype  | 数据类型，可选                              |
| order  | 有"C"和"F"两个选项，分别代表行优先和列优先，在计算机内存中的存储元素的顺序。 |

</font>

---

## 2.1 NumPy简介
- numpy.empty 

```python

import numpy as np 
x = np.empty([3,2], dtype = int) 
print (x)

输出结果为：

[[ 6917529027641081856  5764616291768666155]
 [ 6917529027641081859 -5764598754299804209]
 [          4497473538      844429428932120]]
```

---

## 2.1 NumPy简介

- NumPy 创建数组
  - numpy.zeros创建指定大小的数组，数组元素以 0 来填充：
```python
  numpy.zeros(shape, dtype = float, order = 'C')
```

<font size=5>

| 参数   | 描述                                             |
| -------| ------------------------------------------------ |
| shape  | 数组形状                                         |
| dtype  | 数据类型，可选                                   |
| order  | 'C' 用于 C 的行数组，或者 'F' 用于 FORTRAN 的列数组 |

</font>

---


## 2.1 NumPy简介

- numpy.zeros

```python
import numpy as np
 
x = np.zeros(5) # 默认为浮点数
print(x)
 
y = np.zeros((5,), dtype = int) # 设置类型为整数
print(y)
 
z = np.zeros((2,2), dtype = [('x', 'i4'), ('y', 'i4')])  # 自定义类型
print(z)

输出结果为：

[0. 0. 0. 0. 0.]
[0 0 0 0 0]
[[(0, 0) (0, 0)]
 [(0, 0) (0, 0)]]

```

---


## 2.1 NumPy简介

- NumPy 创建数组
  - numpy.ones 创建指定形状的数组，数组元素以 1 来填充：

```python  
numpy.ones(shape, dtype = None, order = 'C')
```

<font size=5>

| 参数   | 描述                                             |
| -------| ------------------------------------------------ |
| shape  | 数组形状                                         |
| dtype  | 数据类型，可选                                   |
| order  | 'C' 用于 C 的行数组，或者 'F' 用于 FORTRAN 的列数组 |

</font>

---

## 2.1 NumPy简介

- numpy.ones

```python

import numpy as np

x = np.ones(5)  # 默认为浮点数
print(x)

x = np.ones([2,2], dtype = int) # 自定义类型
print(x)

输出结果为：

[1. 1. 1. 1. 1.]
[[1 1]
 [1 1]]

```

--- 


## 2.1 NumPy简介

- NumPy 创建数组
  - numpy.zeros_like 创建一个与给定数组具有相同形状的数组，数组元素以 0 来填充。
  - numpy.zeros 和 numpy.zeros_like 都是用于创建一个指定形状的数组，其中所有元素都是 0。
  - 它们之间的区别在于：numpy.zeros 可以直接指定要创建的数组的形状，而 numpy.zeros_like 则是创建一个与给定数组具有相同形状的数组。
--- 



## 2.1 NumPy简介

- numpy.zeros_like

```python
numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
```

<font size=5>

| 参数    | 描述                                                       |
| ------- | ---------------------------------------------------------- |
| a       | 给定要创建相同形状的数组                                    |
| dtype   | 创建的数组的数据类型                                       |
| order   | 数组在内存中的存储顺序，可选值为 'C'（按行优先）或 'F'（按列优先），默认为 'K'（保留输入数组的存储顺序） |
| subok   | 是否允许返回子类，如果为 True，则返回一个子类对象，否则返回一个与 a 数组具有相同数据类型和存储顺序的数组 |
| shape   | 创建的数组的形状，如果不指定，则默认为 a 数组的形状。       |

</font>

--- 

## 2.1 NumPy简介

- numpy.zeros_like

```python

import numpy as np
 
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 创建一个 3x3 的二维数组
 
zeros_arr = np.zeros_like(arr) # 创建一个与 arr 形状相同的，所有元素都为 0 的数组
print(zeros_arr)

输出结果为：

[[0 0 0]
 [0 0 0]
 [0 0 0]]

```

--- 


## 2.1 NumPy简介

- NumPy 创建数组
  - numpy.ones_like 创建一个与给定数组具有相同形状的数组，数组元素以 1 来填充。
  - numpy.ones 和 numpy.ones_like 都是用于创建一个指定形状的数组，其中所有元素都是 1。
  - 它们之间的区别在于：numpy.ones 可以直接指定要创建的数组的形状，而 numpy.ones_like 则是创建一个与给定数组具有相同形状的数组。
--- 


## 2.1 NumPy简介
- numpy.ones_like 

```python
numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
```

<font size=5>

| 参数    | 描述                                                       |
| ------- | ---------------------------------------------------------- |
| a       | 给定要创建相同形状的数组                                    |
| dtype   | 创建的数组的数据类型                                       |
| order   | 数组在内存中的存储顺序，可选值为 'C'（按行优先）或 'F'（按列优先），默认为 'K'（保留输入数组的存储顺序） |
| subok   | 是否允许返回子类，如果为 True，则返回一个子类对象，否则返回一个与 a 数组具有相同数据类型和存储顺序的数组 |
| shape   | 创建的数组的形状，如果不指定，则默认为 a 数组的形状。       |

</font>

--- 

## 2.1 NumPy简介
- numpy.ones_like 

```python

import numpy as np
 
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 创建一个 3x3 的二维数组
 
ones_arr = np.ones_like(arr) # 创建一个与 arr 形状相同的，所有元素都为 1 的数组
print(ones_arr)

输出结果为：

[[1 1 1]
 [1 1 1]
 [1 1 1]]

```

--- 


## 2.1 NumPy简介

- NumPy 创建数组

<font size=5>

  - numpy.asarray 类似 numpy.array，但 numpy.asarray 参数只有三个，比 numpy.array 少两个。
```
  numpy.asarray(a, dtype = None, order = None)
```

| 参数    | 描述                                                       |
| ------- | ---------------------------------------------------------- |
| a       | 任意形式的输入参数，可以是列表、列表的元组、元组、元组的元组、元组的列表、多维数组等 |
| dtype   | 数据类型，可选                                              |
| order   | 可选，有 "C" 和 "F" 两个选项，分别代表行优先和列优先，在计算机内存中的存储元素的顺序。 |

</font>

--- 

## 2.1 NumPy简介
 - numpy.asarray 将列表转换为 ndarray:

```python

import numpy as np 
 
x =  [1,2,3] 
a = np.asarray(x)  
print (a)

输出结果为：

[1  2  3]
```

--- 


## 2.1 NumPy简介
 - numpy.asarray 将元组转换为 ndarray:

```python

import numpy as np 
 
x =  (1,2,3) 
a = np.asarray(x)  
print (a)

输出结果为：

[1  2  3]
```
--- 


## 2.1 NumPy简介
 - numpy.asarray 将元组列表转换为 ndarray:

```python

import numpy as np 
 
x =  [(1,2,3),(4,5)] 
a = np.asarray(x)  
print (a)
输出结果为：

[(1, 2, 3) (4, 5)]

```
--- 


## 2.1 NumPy简介

- NumPy 创建数组

 - arange 函数创建数值范围并返回 ndarray 对象:
```
 numpy.arange(start, stop, step, dtype)
```
<font size=5> 
 根据 start 与 stop 指定的范围以及 step 设定的步长，生成一个 ndarray。

| 参数    | 描述                                                    |
| ------- | ------------------------------------------------------- |
| start   | 起始值，默认为0                                         |
| stop    | 终止值（不包含）                                        |
| step    | 步长，默认为1                                           |
| dtype   | 返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。 |

</font>

--- 


## 2.1 NumPy简介

- NumPy 创建数组

numpy.arange 生成 0 到 4 长度为 5 的数组:

```python
import numpy as np
 
x = np.arange(5)  
print (x)

输出结果如下：

[0  1  2  3  4]
```

--- 



## 2.1 NumPy简介


- NumPy 创建数组

numpy.arange 设置返回类型位 float:

```python
import numpy as np
 
# 设置了 dtype
x = np.arange(5, dtype =  float)  
print (x)

输出结果如下：

[0.  1.  2.  3.  4.]
```

--- 

## 2.1 NumPy简介

- NumPy 创建数组
<font size=5> 

 - numpy.linspace 函数用于创建一个等差数列构成的一维数组：
```
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

| 参数	| 描述
| ------- | ------------------------------------------------------- |
| start	| 序列的起始值
| stop	| 序列的终止值，如果endpoint为true，该值包含于数列中
| num	| 要生成的等步长的样本数量，默认为50
| endpoint	| 该值为 true 时，数列中包含stop值，反之不包含，默认是True。
| retstep	| 如果为 True 时，生成的数组中会显示间距，反之不显示。
| dtype	| ndarray 的数据类型

</font> 

--- 


## 2.1 NumPy简介

- NumPy 创建数组

np.linspace 设置起始点为 1 ，终止点为 10，数列个数为 10。

```python
import numpy as np
a = np.linspace(1,10,10)
print(a)

输出结果为：

[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
```

--- 


## 2.1 NumPy简介

- NumPy 创建数组

np.linspace 设置元素全部是1的等差数列：

```python
import numpy as np
a = np.linspace(1,1,10)
print(a)

输出结果为：

[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
```

--- 


## 2.1 NumPy简介

- NumPy 创建数组
  
np.linspace  将 endpoint 设为 false，不包含终止值：

```python
import numpy as np

a = np.linspace(10, 20,  5, endpoint =  False)  
print(a)
输出结果为：

[10. 12. 14. 16. 18.]
```

--- 


## 2.1 NumPy简介

- NumPy 创建数组
  
np.linspace 将 endpoint 设为 true，则会包含 20。以下实例设置间距。

```python
import numpy as np
a =np.linspace(1,10,10,retstep= True)
 
print(a)
# 拓展例子
b =np.linspace(1,10,10).reshape([10,1])
print(b)

输出结果为：

(array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]), 1.0)
[[ 1.]
 [ 2.]
 [ 3.]
 [ 4.]
 [ 5.]
 [ 6.]
 [ 7.]
 [ 8.]
 [ 9.]
 [10.]]
```

--- 


## 2.1 NumPy简介

- NumPy 创建数组
  - numpy.logspace函数用于创建一个等比数列:
```
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
```
<font size=4> 
base 参数意思是取对数的时候 log 的下标。

| 参数	| 描述
| ------- | ------------------------------------------------------- |
| start	| 序列的起始值为：base ** start
| stop	| 序列的终止值为：base ** stop。如果endpoint为true，该值包含于数列中
| num	| 要生成的等步长的样本数量，默认为50
| endpoint	| 该值为 true 时，数列中中包含stop值，反之不包含，默认是True。
| base	| 对数 log 的底数。
| dtype	| ndarray 的数据类型

</font> 

--- 

## 2.1 NumPy简介

- NumPy 创建数组

```python
import numpy as np
# np.logspace 默认底数是 10
a = np.logspace(1.0,  2.0, num =  10)  
print (a)

输出结果为：

[ 10.           12.91549665     16.68100537      21.5443469  27.82559402      
  35.93813664   46.41588834     59.94842503      77.42636827    100.    ]
```

---


## 2.1 NumPy简介

- NumPy 创建数组

np.logspace 将对数的底数设置为 2 :

```python
import numpy as np
a = np.logspace(0,9,10,base=2)
print (a)

输出如下：

[  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
```

---





## 2.1 NumPy简介

- NumPy 切片和索引
  - ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。
  - ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。

---


## 2.1 NumPy简介

- NumPy 切片和索引

```python
import numpy as np
 
a = np.arange(10)   # 通过 arange() 函数创建 ndarray 对象
s = slice(2,7,2)   # 从索引 2 开始到索引 7 停止，间隔为2
print (a[s])

输出结果为：

[2  4  6]
```

---


## 2.1 NumPy简介

- NumPy 切片和索引

<font size=5> 
通过冒号分隔切片参数 start:stop:step 来进行切片操作. `冒号:`如果只放置一个参数，如 [2]，将返回与该索引相对应的单个元素。如果为 [2:]，表示从该索引开始以后的所有项都将被提取。如果使用了两个参数，如 [2:7]，那么则提取两个索引(不包括停止索引)之间的项。

```python
import numpy as np
 
a = np.arange(10)  
b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2
print(b)

输出结果为：

[2  4  6]
```
</font> 

---


## 2.1 NumPy简介

- NumPy 切片和索引
 
```python
import numpy as np
 
a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
b = a[5] 
print(b)

输出结果为：

5
```

---



## 2.1 NumPy简介

- NumPy 切片和索引

```python
import numpy as np
 
a = np.arange(10)
print(a[2:])

输出结果为：

[2  3  4  5  6  7  8  9]
```

---



## 2.1 NumPy简介

- NumPy 切片和索引

```python 
import numpy as np
 
a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
print(a[2:5])

输出结果为：

[2  3  4]
```

---


## 2.1 NumPy简介

- NumPy 切片和索引

<font size=5> 

多维数组同样适用上述索引提取方法：

```python 
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
# 从某个索引处开始切割
print('从数组索引 a[1:] 处开始切割')
print(a[1:])

输出结果为：

[[1 2 3]
 [3 4 5]
 [4 5 6]]
从数组索引 a[1:] 处开始切割
[[3 4 5]
 [4 5 6]]
```

</font> 

---



## 2.1 NumPy简介

- NumPy 切片和索引

<font size=5> 

切片还可以包括省略号 …，来使选择元组的长度与数组的维度相同。 如果在行位置使用省略号，它将返回包含行中元素的 ndarray。

```python
import numpy as np
 
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print (a[...,1])   # 第2列元素
print (a[1,...])   # 第2行元素
print (a[...,1:])  # 第2列及剩下的所有元素

输出结果为：

[2 4 5]
[3 4 5]
[[2 3]
 [4 5]
 [5 6]]
```

</font> 

---



## 2.1 NumPy简介

- NumPy 高级索引
  - NumPy 比一般的 Python 序列提供更多的索引方式。
  - 除了之前看到的用整数和切片的索引外，数组可以由整数数组索引、布尔索引及花式索引。
  - NumPy 中的高级索引指的是使用整数数组、布尔数组或者其他序列来访问数组的元素。
  - 相比于基本索引，高级索引可以访问到数组中的`任意元素`，并且可以用来对数组进行复杂的操作和修改。

---


## 2.1 NumPy简介

- NumPy 高级索引
  - 整数数组索引:使用一个数组来访问另一个数组的元素。这个数组中的每个元素都是目标数组中某个维度上的索引值。

```python 
import numpy as np 
 
x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
y = x[[0,1,2],  [0,1,0]]  # 获取数组中 (0,0)，(1,1) 和 (2,0) 位置处的元素。
print (y)

输出结果为：

[1  4  5]
``` 

---

## 2.1 NumPy简介

- NumPy 高级索引
<font size=4> 

```python
import numpy as np 
# 获取 4X3 数组中的四个角的元素。 行索引是 [0,0] 和 [3,3]，而列索引是 [0,2] 和 [0,2]。
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('我们的数组是：' )
print (x)
print ('\n')
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]  
print  ('这个数组的四个角元素是：')
print (y)

我们的数组是：
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

这个数组的四个角元素是：
[[ 0  2]
 [ 9 11]]
``` 

</font> 

---


## 2.1 NumPy简介

- NumPy 广播(Broadcast)
  - 广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。
  - 如果两个数组 a 和 b 形状相同，即满足 a.shape == b.shape，那么 a*b 的结果就是 a 与 b 数组对应位相乘。这要求维数相同，且各维度的长度相同。


---

## 2.1 NumPy简介

- NumPy 广播(Broadcast)

```python
import numpy as np 
 
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print (c)
输出结果为：

[ 10  40  90 160]
```

---

## 2.1 NumPy简介

- NumPy 广播(Broadcast)

<font size=5> 

当运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制。

```python
import numpy as np 
 
a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([0,1,2])
print(a + b)
输出结果为：

[[ 0  1  2]
 [10 11 12]
 [20 21 22]
 [30 31 32]]
```


</font>

---

## 2.1 NumPy简介

- NumPy 广播(Broadcast)

4x3 的二维数组与长为 3 的一维数组相加，等效于把数组 b 在二维上重复 4 次再运算：

 ![width:700](2/2.2.gif)

---

## 2.1 NumPy简介

- NumPy 广播(Broadcast)

<font size=5>
 
```python
import numpy as np 
 
a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([1,2,3])
bb = np.tile(b, (4, 1))  # 重复 b 的各个维度
print(a + bb)
输出结果为：

[[ 1  2  3]
 [11 12 13]
 [21 22 23]
 [31 32 33]]
```

</font>

---

## 2.1 NumPy简介

- NumPy 广播(Broadcast)的规则:
<font size=5> 

  - 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐。
  - 输出数组的形状是输入数组形状的各个维度上的最大值。
  - 如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错。
  - 当输入数组的某个维度的长度为 1 时，沿着此维度运算时都用此维度上的第一组值。
  简单理解：对两个数组，分别比较他们的每一个维度（若其中一个数组没有当前维度则忽略），满足：
    - 数组拥有相同形状。
    - 当前维度的值相等。
    - 当前维度的值有一个是 1。
    - 若条件不满足，抛出 "ValueError: frames are not aligned" 异常。
  
</font>

---


## 2.1 NumPy简介

- NumPy 迭代数组
  - NumPy 迭代器对象 numpy.nditer 提供了一种灵活访问一个或者多个数组元素的方式。

<font size=5> 

```python
import numpy as np
 
a = np.arange(6).reshape(2,3)
print ('原始数组是：')
print (a)
print ('\n')
print ('迭代输出元素：')
for x in np.nditer(a):
    print (x, end=", " )
print ('\n')
```

</font> 

---

## 2.1 NumPy简介

- NumPy 迭代数组

<font size=5> 

```python

输出结果为：

原始数组是：
[[0 1 2]
 [3 4 5]]


迭代输出元素：
0, 1, 2, 3, 4, 5, 
```
- 以上实例不是使用标准 C 或者 Fortran 顺序，选择的顺序是和数组`内存布局`一致的，这样做是为了提升访问的效率，默认是行序优先（row-major order，或者说是 C-order）。
- 默认情况下只需访问每个元素，而无需考虑其特定顺序。

</font> 

---


## 2.1 NumPy简介

- Numpy 数组操作

  - 修改数组形状
  - 翻转数组
  - 修改数组维度
  - 连接数组
  - 分割数组
  - 数组元素的添加与删除
  
---


## 2.1 NumPy简介

- Numpy 修改数组形状

| 函数      | 描述                                   |
|-----------|----------------------------------------|
| reshape   | 不改变数据的条件下修改形状                 |
| flat      | 数组元素迭代器                           |
| flatten   | 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组 |
| ravel     | 返回展开数组                             |

---


## 2.1 NumPy简介

- Numpy 翻转数组

| 函数         | 描述                 |
|--------------|----------------------|
| transpose    | 对换数组的维度         |
| ndarray.T    | 和 self.transpose() 相同 |
| rollaxis     | 向后滚动指定的轴        |
| swapaxes     | 对换数组的两个轴        |


---

## 2.1 NumPy简介

- Numpy 修改数组维度

| 维度           | 描述                       |
|--------------|--------------------------|
| broadcast    | 产生模仿广播的对象            |
| broadcast_to | 将数组广播到新形状             |
| expand_dims  | 扩展数组的形状                |
| squeeze      | 从数组的形状中删除一维条目      |


---

## 2.1 NumPy简介

- Numpy 连接数组

| 函数         | 描述                                 |
|--------------|--------------------------------------|
| concatenate  | 连接沿现有轴的数组序列                |
| stack        | 沿着新的轴加入一系列数组。             |
| hstack       | 水平堆叠序列中的数组（列方向）          |
| vstack       | 竖直堆叠序列中的数组（行方向）          |


---


## 2.1 NumPy简介

- Numpy 分割数组

| 函数    | 数组及操作                             |
|---------|----------------------------------------|
| split   | 将一个数组分割为多个子数组               |
| hsplit  | 将一个数组水平分割为多个子数组（按列）    |
| vsplit  | 将一个数组垂直分割为多个子数组（按行）    |


---


## 2.1 NumPy简介

- Numpy 数组元素的添加与删除

| 函数     | 元素及描述                               |
|---------|----------------------------------------|
| resize  | 返回指定形状的新数组                      |
| append  | 将值添加到数组末尾                        |
| insert  | 沿指定轴将值插入到指定下标之前              |
| delete  | 删掉某个轴的子数组，并返回删除后的新数组      |
| unique  | 查找数组内的唯一元素                      |


---


## 2.1 NumPy简介

<font size=4> 

- Numpy 数学函数

**高性能数组操作**：NumPy 数学函数是针对数组进行优化的，因此它们通常比使用标准 Python 列表的操作快得多。**这是因为 NumPy 使用底层的高效 C 代码来处理数据**。

**支持广播**：NumPy 数学函数支持广播（broadcasting），这意味着它们可以在不同形状的数组之间执行操作，而不需要显式循环。这简化了代码，并提高了性能。

**大量的数学函数**：NumPy 提供了丰富的数学函数，包括基本的算术运算（如加法、减法、乘法、除法）、三角函数、指数和对数函数、线性代数运算、统计函数等。这使得 NumPy 成为科学计算和数据分析的强大工具。

**数组操作**：NumPy 数学函数通常是数组操作，**可以对整个数组或数组的元素执行操作**。这使得它们特别适用于向量化的计算。

**支持多维数组**：NumPy 数学函数支持多维数组，因此可以轻松处理多维数据，例如图像、矩阵和张量。

开源和广泛使用：NumPy 是开源的，广泛用于科学计算、数据分析、机器学习等领域。它构建了许多其他 Python 数据科学库的基础，如 SciPy、Pandas 和 Scikit-Learn。

互操作性：NumPy 数学函数通常可以与其他科学计算库和工具互操作，包括 SciPy、Matplotlib、Pandas、Scikit-Learn 等。这种互操作性使得 NumPy 可以与其他库一起使用，构建完整的数据科学工作流程。


</font> 

---



## 2.1 NumPy简介

- Numpy 数学函数


<font size=2> 

| 函数                   | 描述                                           | 示例                |
|-----------------------|------------------------------------------------|---------------------|
| `numpy.abs(x)`         | 计算数组中各元素的绝对值                       | `numpy.abs([-1, 2, -3])` 返回 `[1, 2, 3]` |
| `numpy.sqrt(x)`        | 计算数组中各元素的平方根                       | `numpy.sqrt([1, 4, 9])` 返回 `[1.0, 2.0, 3.0]` |
| `numpy.square(x)`      | 计算数组中各元素的平方                         | `numpy.square([2, 3, 4])` 返回 `[4, 9, 16]` |
| `numpy.exp(x)`         | 计算数组中各元素的指数值                       | `numpy.exp([0, 1, 2])` 返回 `[1.0, 2.71828183, 7.3890561]` |
| `numpy.log(x)`         | 计算数组中各元素的自然对数值                   | `numpy.log([1, 2, 3])` 返回 `[0.0, 0.69314718, 1.09861229]` |
| `numpy.log10(x)`       | 计算数组中各元素的以10为底的对数值              | `numpy.log10([1, 10, 100])` 返回 `[0.0, 1.0, 2.0]` |
| `numpy.sin(x)`         | 计算数组中各元素的正弦值                       | `numpy.sin([0, numpy.pi/2, numpy.pi])` 返回 `[0.0, 1.0, 0.0]` |
| `numpy.cos(x)`         | 计算数组中各元素的余弦值                       | `numpy.cos([0, numpy.pi/2, numpy.pi])` 返回 `[1.0, 0.0, -1.0]` |
| `numpy.tan(x)`         | 计算数组中各元素的正切值                       | `numpy.tan([0, numpy.pi/4, numpy.pi/2])` 返回 `[0.0, 1.0, inf]` |
| `numpy.arcsin(x)`      | 计算数组中各元素的反正弦值                   | `numpy.arcsin([-1, 0, 1])` 返回 `[-1.57079633, 0.0, 1.57079633]` |
| `numpy.arccos(x)`      | 计算数组中各元素的反余弦值                   | `numpy.arccos([-1, 0, 1])` 返回 `[3.14159265, 1.57079633, 0.0]` |
| `numpy.arctan(x)`      | 计算数组中各元素的反正切值                   | `numpy.arctan([0, 1, -1])` 返回 `[0.0, 0.78539816, -0.78539816]` |
| `numpy.deg2rad(x)`     | 将角度从度数转换为弧度                      | `numpy.deg2rad([0, 90, 180])` 返回 `[0.0, 1.57079633, 3.14159265]` |
| `numpy.rad2deg(x)`     | 将角度从弧度转换为度数                      | `numpy.rad2deg([0, numpy.pi/2, numpy.pi])` 返回 `[0.0, 90.0, 180.0]` |


</font> 

---



## 2.1 NumPy简介

<font size=2> 


| 函数                   | 描述                                   | 示例               |
|-----------------------|----------------------------------------|--------------------|
| `numpy.add(x, y)`      | 将数组中对应元素相加                         | `numpy.add([1, 2, 3], [4, 5, 6])` 返回 `[5, 7, 9]` |
| `numpy.subtract(x, y)` | 将数组中对应元素相减                         | `numpy.subtract([4, 5, 6], [1, 2, 3])` 返回 `[3, 3, 3]` |
| `numpy.multiply(x, y)` | 将数组中对应元素相乘                         | `numpy.multiply([2, 3, 4], [2, 2, 2])` 返回 `[4, 6, 8]` |
| `numpy.divide(x, y)`   | 将数组中对应元素相除                         | `numpy.divide([4, 6, 8], [2, 3, 4])` 返回 `[2.0, 2.0, 2.0]` |


</font> 

---


## 2.1 NumPy简介

NumPy 统计函数

<font size=2> 

| 函数                   | 描述                                   | 示例               |
|-----------------------|----------------------------------------|--------------------|
| `numpy.mean(x)`        | 计算数组中元素的平均值                    | `numpy.mean([1, 2, 3, 4, 5])` 返回 `3.0` |
| `numpy.median(x)`      | 计算数组中元素的中位数                    | `numpy.median([1, 3, 5, 2, 4])` 返回 `3.0` |
| `numpy.var(x)`         | 计算数组中元素的方差                      | `numpy.var([1, 2, 3, 4, 5])` 返回 `2.5` |
| `numpy.std(x)`         | 计算数组中元素的标准差                    | `numpy.std([1, 2, 3, 4, 5])` 返回 `1.58113883` |
| `numpy.sum(x)`         | 计算数组中元素的和                        | `numpy.sum([1, 2, 3, 4, 5])` 返回 `15` |
| `numpy.prod(x)`        | 计算数组中元素的乘积                      | `numpy.prod([1, 2, 3, 4, 5])` 返回 `120` |
| `numpy.min(x)`         | 找出数组中的最小值                       | `numpy.min([1, 2, 3, 4, 5])` 返回 `1` |
| `numpy.max(x)`         | 找出数组中的最大值                       | `numpy.max([1, 2, 3, 4, 5])` 返回 `5` |
| `numpy.argmin(x)`      | 找出最小值的索引                         | `numpy.argmin([1, 2, 3, 4, 5])` 返回 `0` |
| `numpy.argmax(x)`      | 找出最大值的索引                         | `numpy.argmax([1, 2, 3, 4, 5])` 返回 `4` |
| `numpy.percentile(x, p)` | 计算百分位数，其中 p 是百分位数（0到100之间） | `numpy.percentile([1, 2, 3, 4, 5], 25)` 返回 `2.0` |
| `numpy.histogram(x, bins)` | 计算直方图                               | `numpy.histogram([1, 2, 2, 3, 3, 3, 4, 4, 5], bins=[0, 1, 2, 3, 4, 5])` 返回 `(array([1, 2, 3, 2, 1]), array([0, 1, 2, 3, 4, 5]))` |

</font> 

---


## 2.1 NumPy简介

- NumPy 线性代数

<font size=4> 

NumPy 提供了线性代数函数库 linalg，该库包含了线性代数所需的所有功能：

<center> 

| 函数          | 描述                       |
|--------------|--------------------------|
| dot          | 两个数组的点积，即元素对应相乘。     |
| vdot         | 两个向量的叉积。                 |
| cross         | 两个向量的点积。                 |
| inner        | 两个数组的内积。                 |
| matmul       | 两个数组的矩阵积。                |
| multiply       | 两个数组的元素级乘法。                |
| determinant  | 计算数组的行列式。                |
| kron       | 两个张量的 Kronecker 乘积。                |
| tensordot       | 两个张量的点积。                |
| solve        | 求解线性矩阵方程。                |
| inv          | 计算矩阵的乘法逆矩阵。             |

</center> 

</font> 

---

## 2.1 NumPy简介

NumPy 多维数组运用举例: 加法和乘法

<font size=5> 

```python
import numpy as np

# 创建两个2维张量
tensor1 = np.array([[1, 2], [3, 4]])
tensor2 = np.array([[5, 6], [7, 8]])

# 加法
result_add = np.add(tensor1, tensor2)

# 乘法（元素级乘法）
result_multiply = np.multiply(tensor1, tensor2)
```

</font> 


---


## 2.1 NumPy简介

NumPy 多维数组运用举例: 线性代数

<font size=3> 

```python
import numpy as np

# 创建两个2维张量
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# 矩阵乘法
matrix_multiply = np.dot(matrix_a, matrix_b)

# 矩阵转置
matrix_transpose = np.transpose(matrix_a)

# 行列式计算
matrix_det = np.linalg.det(matrix_a)

# 逆矩阵计算
matrix_inv = np.linalg.inv(matrix_a)

# 特征值和特征向量计算
eigenvalues, eigenvectors = np.linalg.eig(matrix_a)

# 打印结果
print("Matrix A:\n", matrix_a)
print("Matrix B:\n", matrix_b)
print("Matrix Multiplication:\n", matrix_multiply)
print("Matrix Transpose:\n", matrix_transpose)
print("Matrix Determinant:\n", matrix_det)
print("Matrix Inverse:\n", matrix_inv)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

```

</font> 


---



## 2.1 NumPy简介

NumPy 多维数组运用举例: 索引和切片

<font size=5> 

```python
import numpy as np

# 创建一个3维张量
tensor3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 获取特定元素
element = tensor3D[1, 0, 1]  # 获取第二个 "层" 的第一个 "行" 的第二个元素

# 切片
slice_tensor = tensor3D[:, 1, :]  # 获取所有 "层" 的第二个 "行"

```

</font> 


---


## 2.1 NumPy简介

NumPy 多维数组运用举例: 统计函数

<font size=5> 

```python
import numpy as np

# 创建一个2维张量
tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 求和
sum_tensor = np.sum(tensor)

# 沿特定轴计算平均值
mean_along_axis = np.mean(tensor, axis=0)

# 沿特定轴计算标准差
std_deviation_along_axis = np.std(tensor, axis=1)
```

</font> 


---


## 2.1 NumPy简介

NumPy 多维数组运用举例: 非线性运算

<font size=3> 

```python
import numpy as np

# 创建一个2维张量
tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 非线性运算 - 平方根
sqrt_result = np.sqrt(tensor)

# 非线性运算 - 指数运算
exp_result = np.exp(tensor)

# 非线性运算 - 对数运算
log_result = np.log(tensor)

# 非线性运算 - 正弦函数
sin_result = np.sin(tensor)

# 非线性运算 - 双曲正弦函数
sinh_result = np.sinh(tensor)

# 非线性运算 - 阶乘
factorial_result = np.math.factorial(tensor)

# 打印结果
print("原始张量:\n", tensor)
print("平方根:\n", sqrt_result)
print("指数运算:\n", exp_result)
print("对数运算:\n", log_result)
print("正弦函数:\n", sin_result)
print("双曲正弦函数:\n", sinh_result)
print("阶乘:\n", factorial_result)

```

</font> 


---


## 2.1 NumPy简介

NumPy 多维数组运用举例: 高维张量不支持所有的线性代数运算


<font size=3> 

```python
import numpy as np

# 创建两个4维张量
tensor_a = np.random.rand(2, 2, 3, 3)  # 2x2x3x3张量
tensor_b = np.random.rand(2, 2, 3, 3)  # 2x2x3x3张量

# 矩阵乘法
tensor_multiply = np.matmul(tensor_a, tensor_b)

# 矩阵转置
tensor_transpose = np.transpose(tensor_a, axes=(0, 1, 3, 2))  # 交换第三和第四维度

# 行列式计算
# 注意：4维张量的行列式通常没有直接的数学定义
# 因此，此示例仅为演示
# 通常需要特定问题背景下的行列式计算
# tensor_det = np.linalg.det(tensor_a)  # 这将引发异常

# 逆矩阵计算
# 同样，逆矩阵在4维张量中也需要特定的问题背景
# tensor_inv = np.linalg.inv(tensor_a)  # 这将引发异常

# 打印结果
print("Tensor A:\n", tensor_a)
print("Tensor B:\n", tensor_b)
print("Tensor Multiplication:\n", tensor_multiply)
print("Tensor Transpose:\n", tensor_transpose)
```

</font> 


---




## 2.2 △ Pandas简介

- Anaconda虚拟环境下使用pip安装pandas
```python
  pip install pandas
```

导入 pandas 一般使用别名 pd 来代替：
```python
  import pandas as pd
```

---  

## 2.2 △ Pandas简介
- Pandas 数据结构
  1. Series
  2. DataFrame


---  

## 2.2 △ Pandas简介
- Pandas 数据结构- Series
<font size=5> 

  Pandas Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型。 Series 由索引（index）和列组成：
```python
  pandas.Series(data, index, dtype, name, copy)
```
| 参数   | 描述                                     | 默认值    |
|--------|------------------------------------------|-----------|
| data   | 一组数据(ndarray 类型)                   | 无       |
| index  | 数据索引标签，如果不指定，默认从 0 开始   | 0, 1, 2, ... |
| dtype  | 数据类型， 默认会自己判断                | 自动判断  |
| name   | 设置名称                                | 无       |
| copy   | 拷贝数据，默认为 False                   | False     |

</font> 


---  

## 2.2 △ Pandas简介
- Pandas 数据结构- Series

<font size=5>

```python
import pandas as pd

a = [1, 2, 3]

myvar = pd.Series(a) # 创建一个简单的 Series

print(myvar)

```

输出:

</font> 

![width:300](2/2.3.png)  

---


## 2.2 △ Pandas简介
- Pandas 数据结构- Series
<font size=5>

如果没有指定索引，索引值就从 0 开始

```python
import pandas as pd

a = [1, 2, 3]

myvar = pd.Series(a)

print(myvar[1]) # 可以根据索引值读取数据
```

输出：

</font> 

2


---


## 2.2 △ Pandas简介
- Pandas 数据结构- Series
<font size=5>

```python
import pandas as pd

a = ["Google", "Runoob", "Wiki"]

myvar = pd.Series(a, index = ["x", "y", "z"]) #  指定索引值

print(myvar)

```
输出:

</font> 

![width:300](2/2.4.jpg)

---


## 2.2 △ Pandas简介
- Pandas 数据结构- Series
<font size=5>

```python
import pandas as pd

a = ["Google", "Runoob", "Wiki"]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar["y"])   #  根据索引值读取数据

```
输出:

</font> 

Runoob

---


## 2.2 △ Pandas简介
- Pandas 数据结构- Series
<font size=5>

可以使用 key/value 对象，类似字典来创建 Series：

```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"} # 字典的 key 变成了索引值

myvar = pd.Series(sites)

print(myvar)

输出:

1    Google
2    Runoob
3      Wiki
dtype: object
```

</font> 

---


## 2.2 △ Pandas简介
- Pandas 数据结构- Series
<font size=5>

如果只需要字典中的一部分数据，只需要指定需要数据的索引即可:

```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}

myvar = pd.Series(sites, index = [1, 2])

print(myvar)

输出:

1    Google
2    Runoob
dtype: object
```

</font> 

---



## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame
  - DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
  - Pandas DataFrame 是一个二维的数组结构，类似二维数组。
---


## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame

![width:500](2/2.5.png)

![width:500](2/2.6.png)

---


## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame
<font size=5>

DataFrame 构造方法如下：
```python
pandas.DataFrame( data, index, columns, dtype, copy)
```

| 参数      | 描述                                       | 默认值    |
|-----------|--------------------------------------------|-----------|
| data      | 一组数据(ndarray、series, map, lists, dict等类型) | 无       |
| index     | 索引值，或者可以称为行标签                 | 无       |
| columns   | 列标签，默认为 RangeIndex (0, 1, 2, …, n)   | RangeIndex |
| dtype     | 数据类型                                   | 自动判断  |
| copy      | 拷贝数据，默认为 False                    | False     |

</font> 

---


## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame
<font size=5>

使用列表创建
```python
import pandas as pd

data = [['Google',10],['Runoob',12],['Wiki',13]]

df = pd.DataFrame(data,columns=['Site','Age'],dtype=float)

print(df)
```

输出:

```
     Site   Age
0  Google  10.0
1  Runoob  12.0
2    Wiki  13.0
```

</font> 

---


## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame

<font size=5>

使用 ndarrays 创建，ndarray 的长度必须相同， 如果传递了 index，则索引的长度应等于数组的长度。如果没有传递索引，则默认情况下，索引将是range(n)，其中n是数组长度。

```python
import pandas as pd

data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}

df = pd.DataFrame(data)

print (df)
```

输出结果如下：
```
     Site  Age
0  Google   10
1  Runoob   12
2    Wiki   13
```

</font> 

---


## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame

从以上输出结果可以知道， DataFrame 数据类型一个表格，包含 rows（行） 和 columns（列）：

![width:500](2/2.7.png)

---


## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame

<font size=5>

可以使用字典（key/value），其中字典的 key 为列名:
```python
import pandas as pd

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

df = pd.DataFrame(data)

print (df)
```
输出结果为：
```
   a   b     c
0  1   2   NaN   # 没有对应的部分数据为 NaN。
1  5  10  20.0
```

</font> 

---


## 2.2 △ Pandas简介
- Pandas 数据结构 - DataFrame

<font size=5>

Pandas 可以使用 loc 属性返回指定行的数据，如果没有设置索引，第一行索引为 0，第二行索引为 1，以此类推：

```python
import pandas as pd
data = {"calories": [420, 380, 390], "duration": [50, 40, 45]}
df = pd.DataFrame(data)  # 数据载入到 DataFrame 对象
print(df.loc[0])  # 返回第一行：返回结果其实就是一个 Pandas Series 数据。
print(df.loc[1])   # 返回第二行：返回结果其实就是一个 Pandas Series 数据。
```
输出：
```
calories    420
duration     50
Name: 0, dtype: int64
calories    380
duration     40
Name: 1, dtype: int64
```

</font> 

---


<font size=5>

可以返回多行数据，使用 [[ ... ]] 格式，... 为各行的索引，以逗号隔开：

```python
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)

# 返回第一行和第二行： 返回结果其实就是一个 Pandas DataFrame 数据。
print(df.loc[[0, 1]])
```
输出：
```
   calories  duration
0       420        50
1       380        40
```


</font> 

---



<font size=5>

可以指定索引值，如下实例：
```python
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

print(df)

```
输出结果为：
```
      calories  duration
day1       420        50
day2       380        40
day3       390        45
```

</font> 

---


<font size=5>

Pandas 可以使用 loc 属性返回指定索引对应到某一行：
```python
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

# 指定索引
print(df.loc["day2"])
```

输出：
```
calories    380
duration     40
Name: day2, dtype: int64
```

</font> 

---


## 2.2 △ Pandas简介
- Pandas 数据结构 CSV 文件
<font size=5>

  - CSV（Comma-Separated Values，逗号分隔值，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本）。

  - CSV 是一种通用的、相对简单的文件格式，被用户、商业和科学广泛应用。

</font> 

```python
import pandas as pd

df = pd.read_csv('nba.csv')

print(df.to_string()) # to_string() 用于返回 DataFrame 类型的数据
# 如果不使用该函数，则输出结果为数据的前面 5 行和末尾 5 行，中间部分以 ... 代替。
```

---

## 2.2 △ Pandas简介
- Pandas 数据结构 CSV 文件
```python
import pandas as pd

df = pd.read_csv('nba.csv')

print(df)
```

输出结果为：

<font size=5>

```

              Name            Team  Number Position   Age Height  Weight            College     Salary
0    Avery Bradley  Boston Celtics     0.0       PG  25.0    6-2   180.0              Texas  7730337.0
1      Jae Crowder  Boston Celtics    99.0       SF  25.0    6-6   235.0          Marquette  6796117.0
2     John Holland  Boston Celtics    30.0       SG  27.0    6-5   205.0  Boston University        NaN
3      R.J. Hunter  Boston Celtics    28.0       SG  22.0    6-5   185.0      Georgia State  1148640.0
4    Jonas Jerebko  Boston Celtics     8.0       PF  29.0   6-10   231.0                NaN  5000000.0
..             ...             ...     ...      ...   ...    ...     ...                ...        ...
453   Shelvin Mack       Utah Jazz     8.0       PG  26.0    6-3   203.0             Butler  2433333.0
454      Raul Neto       Utah Jazz    25.0       PG  24.0    6-1   179.0                NaN   900000.0
455   Tibor Pleiss       Utah Jazz    21.0        C  26.0    7-3   256.0                NaN  2900000.0
456    Jeff Withey       Utah Jazz    24.0        C  26.0    7-0   231.0             Kansas   947276.0
457            NaN             NaN     NaN      NaN   NaN    NaN     NaN                NaN        NaN
```
</font> 


---

## 2.2 △ Pandas简介
- Pandas 数据结构 CSV 文件
<font size=5>
可以使用 to_csv() 方法将 DataFrame 存储为 csv 文件：

```python

import pandas as pd
   
# 三个字段 name, site, age
nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]
   
# 字典
dict = {'name': nme, 'site': st, 'age': ag}
     
df = pd.DataFrame(dict)
 
# 保存 dataframe
df.to_csv('site.csv')
```

</font> 

---


## 2.2 △ Pandas简介
<font size=3>
- Pandas 数据结构 CSV 文件 数据处理方法

| 函数                          | 描述                                                  | 示例                             |
|------------------------------|-------------------------------------------------------|----------------------------------|
| `read_csv(filepath_or_buffer)` | 从CSV文件中读取数据并返回一个DataFrame对象           | `pd.read_csv('data.csv')`         |
| `to_csv(path_or_buf)`         | 将DataFrame数据写入CSV文件                           | `df.to_csv('output.csv')`         |
| `read_excel(io)`              | 从Excel文件中读取数据并返回一个DataFrame对象         | `pd.read_excel('data.xlsx')`      |
| `to_excel(excel_writer)`      | 将DataFrame数据写入Excel文件                         | `df.to_excel('output.xlsx')`      |
| `head(n)`                     | 返回前n行数据                                          | `df.head(5)`                     |
| `tail(n)`                     | 返回后n行数据                                          | `df.tail(5)`                     |
| `info()`                      | 显示DataFrame的基本信息，包括数据类型和缺失值情况    | `df.info()`                      |
| `describe()`                  | 生成描述性统计信息，如均值、标准差、最大值、最小值等 | `df.describe()`                  |
| `shape`                       | 返回DataFrame的维度，即行数和列数                    | `df.shape`                        |
| `columns`                     | 返回DataFrame的列标签                                | `df.columns`                      


</font> 

---

<font size=3>

| 函数                          | 描述                                                  | 示例                             |
|------------------------------|-------------------------------------------------------|----------------------------------|
|
| `index`                       | 返回DataFrame的索引标签                              | `df.index`                        |
| `loc[]`                       | 通过标签选择行和列数据                               | `df.loc[2, 'column_name']`        |
| `iloc[]`                      | 通过整数位置选择行和列数据                           | `df.iloc[2, 1]`                   |
| `drop(labels, axis)`          | 删除行或列，`axis`为0表示删除行，为1表示删除列         | `df.drop(2, axis=0)`              |
| `fillna(value)`               | 填充缺失值，使用指定的值或方法                        | `df.fillna(0)`                   |
| `dropna()`                    | 删除包含缺失值的行                                   | `df.dropna()`                    |
| `groupby(by)`                 | 对数据进行分组，通常与聚合函数一起使用                | `df.groupby('column_name')`       |
| `agg()`                       | 对分组后的数据进行聚合操作                           | `df.groupby('column_name').agg({'column_name': 'sum'})` |
| `pivot_table()`               | 创建数据透视表                                       | `pd.pivot_table(df, values='value', index='index', columns='column')` |
| `sort_values(by)`             | 根据指定列的值对数据进行排序                         | `df.sort_values(by='column_name')` |
| `merge(right)`                | 合并两个DataFrame，类似SQL的JOIN操作                 | `pd.merge(df1, df2, on='key')`    |
| `corr()`                      | 计算列之间的相关性矩阵                               | `df.corr()`                      |
| `plot()`                      | 生成数据可视化图表                                   | `df.plot(kind='bar', x='index', y='value')` |


</font> 

---


## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

**openpyxl** 是一个用于处理 Excel 文件（.xlsx 格式）的 Python 库。使用 openpyxl，可以创建自动化的 Excel 处理工具、数据报告生成器、数据清洗工具等。

```
pip install openpyxl
```

openpyxl 的主要特点和用途：

**创建 Excel 文件**：openpyxl 允许创建新的 Excel 工作簿和工作表，然后向其添加数据和样式。

**读取 Excel 文件**：可以使用 openpyxl 读取现有的 Excel 文件，包括工作表中的数据、样式、公式等。

**修改 Excel 文件**：可以使用 openpyxl 修改现有的 Excel 文件，包括更新数据、样式、插入/删除行列等。

**处理图表**：openpyxl 允许处理 Excel 中的图表，包括创建、修改和删除图表。

**自定义样式**：可以设置单元格的字体、颜色、边框等样式，以创建具有专业外观的工作表。


</font> 

---


## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

**支持公式**：openpyxl 可以处理工作表中的公式，包括计算公式结果和更新公式。

**数据验证**：可以添加数据验证规则，以确保工作表中的数据符合特定的规则和约束。

**支持图像和注释**：可以插入图像和注释来更好地解释工作表中的数据。

**兼容性**：openpyxl 支持 Excel 2010 及更高版本的文件格式。

**开源和广泛使用**：openpyxl 是一个开源项目，受到广泛支持和社区贡献。


</font> 

---


## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

**pd.read_excel()** 是 pandas 库中用于从 Excel 文件中读取数据的函数（ 需要 openpyxl 支持 ）。并将 Excel 文件转换为 pandas 数据框，以便进行数据分析和处理。以下是关于函数的参数和用途的介绍：

参数说明：
**io**	Excel 文件的路径或可用于读取数据的对象。
**sheet_name**	要读取的工作表名称，可以是字符串、整数或None。
**header**	指定包含列名的行号，默认为0。
**index_col**	用作行索引的列名或列号，默认为None。
**usecols**	指定要读取的列，可以是列名或列号的列表。
**skiprows**	跳过指定的行数后再读取数据。
**nrows**	读取的行数。
**na_values**	指定用于表示缺失数据的值。
**parse_dates**	解析日期列，默认为False。
**date_parser**	用于解析日期列的函数。
**dtype**	指定每列的数据类型。
**converters**	自定义列值的转换函数。
**skipfooter**	跳过文件末尾的行数。

</font> 

---




## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>


```python
import pandas as pd

df = pd.read_excel('成绩单.xlsx', sheet_name='Sheet1')

print(df.head())



输出：

      学号  姓名  语文  数学  英语
0  10001  赵一  85  85  85
1  10002  钱二  52  52  52
2  10003  张三  12  12  12
3  10004  李四  21  21  21
4  10005  王五  88  88  88

```


</font> 

---




## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

```python
import pandas as pd

df = pd.read_excel('成绩单.xlsx', sheet_name='Sheet1', index_col=0)

print(df.head())



输出：

       姓名  语文  数学  英语
学号                   
10001  赵一  85  85  85
10002  钱二  52  52  52
10003  张三  12  12  12
10004  李四  21  21  21
10005  王五  88  88  88


```


</font> 

---



## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

```python
import pandas as pd

df = pd.read_excel('成绩单.xlsx', sheet_name='Sheet2')

print(df.head())



输出：

       工号   姓名   部门  法定年假时数(H)  已休年假（H)  待休年假(H)
0  A00004   张四  总务部         60       14       46
1  A00013   李四  总务部         60       29       31
2  A00022   王四  总务部         40       28       12
3  A00031  王十三  总务部         40        0       40
4  A00040   刘三  总务部         40       14       26


```


</font> 

---




## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

```python
import pandas as pd

df = pd.read_excel('成绩单.xlsx', sheet_name='Sheet1', index_col=0, dtype={'数学': float})

print(df.head())



输出：

       姓名  语文    数学  英语
学号                     
10001  赵一  85  85.0  85
10002  钱二  52  52.0  52
10003  张三  12  12.0  12
10004  李四  21  21.0  21
10005  王五  88  88.0  88


```

</font> 

---


## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

```python
import pandas as pd

df = pd.read_excel('成绩单.xlsx', sheet_name='Sheet1', index_col=0, na_values={'姓名':"赵一"})


print(df.head())



输出：

        姓名  语文  数学  英语
学号                    
10001  NaN  85  85  85
10002   钱二  52  52  52
10003   张三  12  12  12
10004   李四  21  21  21
10005   王五  88  88  88

```

</font> 

---



## 2.2 △ Pandas简介
- Pandas 读取Excel 文件
<font size=4>

```python

import pandas as pd

# 读取 Excel 文件，指定工作表名、列名行、数据类型、缺失值标识等参数
df = pd.read_excel('example.xlsx',
                   sheet_name='Sheet1',
                   header=1,  # 列名在第二行
                   names=['Name', 'Age', 'Grade'],  # 自定义列名
                   index_col='Name',  # 使用 'Name' 列作为行索引
                   usecols=['Name', 'Age'],  # 只读取 'Name' 和 'Age' 列
                   dtype={'Age': int},  # 指定 'Age' 列的数据类型
                   converters={'Name': str.upper},  # 将 'Name' 列的值转换为大写
                   true_values=['Yes'],  # 指定 'Yes' 为 True
                   false_values=['No'],  # 指定 'No' 为 False
                   keep_default_na=False,  # 不保留默认的缺失值标识
                   na_values=['NA', 'N/A'],  # 指定 'NA' 和 'N/A' 为缺失值
                   skiprows=[2],  # 跳过第三行
                   nrows=3  # 读取前三行数据
                   )

# 打印读取的数据框
print(df)
```

</font> 

---


## 2.3 数据获取

<font size=4>

数据进行清洗、处理和分析之前需要获取行业数据。Python提供了多种方法要获取行业数据，包括使用API、网络爬虫和数据库连接。以下是一些通用的方法：

1. **使用数据API**：许多行业和数据提供商提供API，允许开发者访问其数据。

2. **网络爬虫**：您可以使用Python编写网络爬虫来从网站上抓取数据。使用库如Beautiful Soup和Scrapy可以更轻松地进行网页抓取。请确保遵守网站的使用政策和法律法规。

3. **数据库连接**：如果行业数据存储在数据库中，您可以使用Python库（如SQLAlchemy、psycopg2等）来连接数据库，并执行SQL查询以检索数据。

4. **开源数据集**：许多行业和领域有可用的开源数据集。您可以在数据科学和机器学习竞赛平台（如Kaggle）上找到这些数据集，或者使用Python库（如Pandas）来加载和分析它们。

5. **数据采集工具**：一些数据采集工具，如Selenium，可以模拟用户操作浏览器，以获取特定网站上的数据。

6. **专业数据提供商**：有些行业数据可能需要购买，而不是免费获取。专业数据提供商通常提供订阅服务，以获得高质量和实时数据。

**无论选择哪种方法，都要确保遵守数据使用政策和法律法规，以确保数据的合法和合规使用。**

</font> 

---



## 2.3 数据获取

<font size=4>

**行业数据API**


Python可以使用各种行业数据API来获取不同领域的信息。以下是一些常见的行业数据API和它们可以提供的信息。这些API通常需要注册和获取API密钥，以便在Python应用程序中进行访问。开发者可以根据其特定需求选择合适的API，并查看官方文档以了解如何使用它们。一些API可能会有使用限制和费用，因此需要仔细研究它们的使用条款。

**1. 金融数据API**：

在国内，开发者可以申请并使用一些国际上免费提供的金融数据API，这些API通常提供了广泛的金融市场数据，包括股票、外汇、加密货币等。以下是一些国际上常见的免费金融数据API，可以在国内使用：

**Alpha Vantage**：Alpha Vantage提供了丰富的金融数据，包括股票、外汇和加密货币。他们提供了免费的API访问，同时也有一些高级功能需要付费。

**Yahoo Finance**：Yahoo Finance提供了股票市场数据的免费API，包括实时行情、历史数据和财务指标。这是一个常用的金融数据来源。

**IEX Cloud**：IEX Cloud提供了股票市场数据的免费API，包括美国股票市场的数据。他们也提供了一些高级功能需要付费。

**Quandl**：Quandl提供了广泛的金融和经济数据，包括股票、期货、外汇、经济指标等。他们有一个免费的API层，同时也提供了付费订阅。

**Open Exchange Rates**：Open Exchange Rates提供了外汇汇率数据的免费API，包括实时汇率和历史数据。

**FRED (Federal Reserve Economic Data)**：FRED由美国联邦储备系统提供，提供了经济数据的免费API，包括经济指标和数据系列。


</font> 

---

## 2.3 数据获取

<font size=4>

在国内，有一些免费的金融数据API可以供开发者申请和使用。以下是一些国内常见的免费金融数据API：

**新浪财经API**：新浪财经提供了一些免费的金融数据API，包括股票、基金、期货等数据。开发者可以通过申请API Key来获取数据。

**聚宽数据API**：聚宽是一家提供金融数据服务的公司，他们提供了一些基本的免费API，包括A股股票数据和基金数据。需要注册并获取API Token。

**米筐数据API**：米筐数据提供了一些A股股票和基金的免费API，包括实时行情和历史数据。开发者可以在其官网注册并获取API Key。

**雪球API**：雪球是一家专注于股票数据的社交平台，他们提供了一些免费的股票数据API，包括实时行情和历史数据。

**天天基金网API**：天天基金网提供了一些免费的基金数据API，包括基金净值、基金排行等。需要注册并获取API Key。

</font> 

---



## 2.3 数据获取

<font size=4>


**Alpha Vantage API** 提供了多种功能，主要用于获取实时和历史股票市场数据以及相关技术指标。以下是 Alpha Vantage API 的主要功能：

**实时股票数据**：您可以使用 Alpha Vantage API 获取实时股票价格、交易量和相关数据。这使您能够监控当前市场情况。

**历史股票数据**：API允许您检索特定日期范围内的历史股票数据，包括每日开盘价、收盘价、最高价和最低价。

**技术指标**：Alpha Vantage提供了超过50个技术指标，如移动平均线、相对强度指数（RSI）、布林带等，以帮助您分析股票走势。

**股票搜索和元数据**：您可以搜索特定股票的元数据，包括股票名称、交易所信息和市场信息。

**外汇数据**：API还支持获取外汇市场数据，包括不同货币对的实时价格。

**加密货币数据**：Alpha Vantage还提供了加密货币市场数据，允许您追踪比特币等加密货币的价格。

**全球股票指数**：API支持获取全球股票指数的数据，包括标普500、道琼斯工业平均指数等。

**分析工具**：Alpha Vantage还提供了分析工具，如技术指标计算器和股票策略回测，以帮助投资者更好地分析股票和市场。



</font> 

---

## 2.3 数据获取

<font size=4>


**Alpha Vantage API**

```python
import requests

# 替换成您的API密钥
api_key = 'T4ESMGDOV2YAMRUO' # 163


# 股票代码，例如苹果公司
symbol = 'AAPL'

# 构建API请求URL
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}'

# 发送API请求
response = requests.get(url)

# 解析JSON响应
data = response.json()

# 提取实时股票价格
latest_price = data['Time Series (1min)'][list(data['Time Series (1min)'].keys())[0]]['1. open']

print(f'The latest price of {symbol} is {latest_price}')

输出：

The latest price of AAPL is 168.0110

```

</font> 

---




## 2.3 数据获取

<font size=4>

**2.天气数据API**：

**OpenWeatherMap**：OpenWeatherMap提供了一个免费的API，允许您获取当前天气、未来预报和历史天气数据。您需要注册一个帐户，然后可以使用他们的API密钥来访问数据。

**Weatherstack**：Weatherstack也提供一个免费的API，可以获取全球范围内的天气数据，包括实时和预报数据。您需要注册并获取API密钥。

**ClimaCell (Now Tomorrow.io)**：ClimaCell提供了一些有关天气和气象的数据，他们有一个免费的开发者计划，可以使用API密钥来获取数据。

**Weatherbit**：Weatherbit提供一个免费的天气API，可以获取全球范围内的实时天气和预报数据。注册后，您可以获取API密钥。

**Met Office (英国气象局)**：Met Office提供了英国的天气数据，他们的API也包括免费访问级别。您需要注册并获得API密钥。

**National Weather Service (美国国家气象局)**：美国国家气象局提供了免费的API，可以获取美国境内的天气数据。您可以注册并获得API密钥。

</font> 

---



## 2.3 数据获取

<font size=4>

**2.天气数据API**：

**和风天气**：和风天气提供了免费的API，允许您获取实时天气、未来预报、空气质量和其他气象数据。您需要注册一个帐户，然后可以获取API密钥。

**心知天气**：心知天气也提供了免费的天气API，可以获取全球范围内的实时天气和预报数据。您需要注册并获取API密钥。

**中国天气网**：中国天气网提供了一些免费的天气API，包括城市天气、气象预报和空气质量数据。您需要注册并获得API密钥。

**气象数据云**：气象数据云是中国气象局推出的免费天气数据服务，可以获取中国境内的各种气象数据。您需要注册并获取API密钥。

**阿里云天气开放平台**：阿里云也提供了免费的天气API，允许您获取全球范围内的天气数据。您需要注册并获取API密钥。

</font> 

---



## 2.3 数据获取

<font size=4>


**3. 地理数据API**：

Google Maps API: 提供地理位置数据，包括地图、地理编码、路径规划等功能。
Mapbox API: 提供自定义地图和地理信息数据，用于地图可视化和分析。



**4. 社交媒体数据API**：

Twitter API: 允许访问Twitter上的实时社交媒体数据，包括推文、用户信息等。
Facebook Graph API: 提供对Facebook社交媒体平台数据的访问权限。


**5. 健康数据API**：

Fitbit API: 提供健康和运动数据，包括步数、心率、睡眠等信息。
MyFitnessPal API: 允许访问健身和饮食数据。


**6. 新闻数据API**：

News API: 提供新闻文章和头条新闻的数据，包括各种新闻来源。
New York Times API: 提供对《纽约时报》新闻文章的访问权限。


**7. 电商数据API**：

Amazon Product Advertising API: 提供对亚马逊网站上产品信息的访问权限，包括价格、评论等。



</font> 

---



## 2.3 数据获取

<font size=4>

Python开源数据集:

Python拥有许多开源数据集，适用于各种应用领域，包括机器学习、数据分析和数据可视化等。以下是一些常见的Python开源数据集，涵盖不同领域：

**Iris数据集**：经典的机器学习数据集，包含三个不同种类的鸢尾花的测量数据，常用于分类问题。

**MNIST手写数字数据集**：包含大量手写数字的图像，用于图像识别和深度学习模型的训练。

**CIFAR-10和CIFAR-100数据集**：包含小图像的数据集，适用于图像分类和对象识别任务。

**Titanic数据集**：关于泰坦尼克号船上乘客的信息，可用于生存预测问题。

**IMDB电影评论数据集**：包含电影评论和情感标签，用于情感分析任务。

**Wine数据集**：关于不同种类的葡萄酒的化学分析数据，用于分类问题。

**Boston Housing数据集**：包含波士顿地区房屋价格的数据，用于回归问题。

**Fashion MNIST**：类似于MNIST，但包含时尚商品的图像，用于图像分类任务。


</font> 

---


## 2.3 数据获取

<font size=4>


**Yelp评论数据集**：包含用户对商家的评论和评级，用于自然语言处理任务。

**COVID-19数据集**：包含关于COVID-19疫情的数据，可用于疫情分析和可视化。

**UCI机器学习库**：UCI机器学习库提供了大量机器学习数据集，涵盖多个领域。

**OpenAI GPT-3 Playground**：用于生成文本的数据集，可用于自然语言生成任务。


这些数据集通常可以通过Python库（如scikit-learn、TensorFlow、PyTorch等）或在线资源进行访问和下载。根据具体需求，可以选择合适的数据集用于项目或研究。


</font> 

---




## 2.4 数据清洗和预处理
- Pandas 数据清洗
  - 数据清洗是对一些没有用的数据进行处理的过程。
  - 很多数据集存在数据缺失、数据格式错误、错误数据或重复数据的情况，如果要使数据分析更加准确，就需要对这些没有用的数据进行处理。
---

## 2.4 数据清洗和预处理
- Pandas 数据清洗

<font size=5>

| 函数                             | 说明                                   |
|----------------------------------|----------------------------------------|
| `df.dropna()`                    | 删除包含缺失值的行或列                  |
| `df.fillna(value)`               | 将缺失值替换为指定的值                |
| `df.replace(old_value, new_value)` | 将指定值替换为新值                   |
| `df.duplicated()`                | 检查是否有重复的数据                  |
| `df.drop_duplicates()`           | 删除重复的数据                         |

</font> 

---



## 2.4 数据清洗和预处理

<font size=5>

**可处理的文件类型:**

1. excel
2. word
3. ppt
4. pdf
5. mail
6. 文件_目录

</font> 

---


## 2.4 数据清洗和预处理

<font size=5>

**自动化办公库(第三方库)：**

xlwings
openpyxl
xlrd
xlwt
xlutils
xlsxwriter
pandas
python-docx
python-pptx
PyPDF2
NumPy
email



</font> 

---




## 2.5 △ 数据可视化（Matplotlib）

- Anaconda虚拟环境下使用pip安装Matplotlib

```python
  pip install Matplotlib
```
- 导入 matplotlib 库，然后查看 matplotlib 库的版本号：
```python
  import matplotlib

  print(matplotlib.__version__)
```

---


## 2.5 △ 数据可视化（Matplotlib）
- Matplotlib Pyplot

  - Pyplot 是 Matplotlib 的子库，提供了和`MATLAB`类似的绘图 API。

  - Pyplot 是常用的绘图模块，能很方便让用户绘制 2D 图表。

  - Pyplot 包含一系列绘图函数的相关函数，每个函数会对当前的图像进行一些修改，例如：给图像加上标记，生新的图像，在图像中产生新的绘图区域等等。

---  


## 2.5 △ 数据可视化（Matplotlib）
- Matplotlib Pyplot 常用函数

<font size=5>

| 函数           | 说明                           |
|----------------|--------------------------------|
| `plot()`       | 用于绘制线图和散点图           |
| `scatter()`    | 用于绘制散点图                 |
| `bar()`        | 用于绘制垂直条形图和水平条形图 |
| `hist()`       | 用于绘制直方图                 |
| `pie()`        | 用于绘制饼图                   |
| `imshow()`     | 用于绘制图像                   |
| `subplots()`   | 用于创建子图                   |

</font>

---  



- Matplotlib 图的组件

<center>

![width:400](anatomy.jpg)

</center>

<font size=4>



</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

例1：柱状图

```python

import matplotlib.pyplot as plt

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 定义水果名称和数量
fruits = ['苹果', '蓝莓', '樱桃', '橙子']
counts = [40, 100, 30, 55]

# 设置柱状图的标签和颜色
bar_labels = ['红色', '蓝色', '红色', '橙色']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

# 创建柱状图
ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

# 设置Y轴标签和图表标题
ax.set_ylabel('水果供应量')
ax.set_title('不同种类和颜色的水果供应')
ax.legend(title='水果颜色')

# 显示图表
plt.show()



```



</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

<center>

![Alt text](Figure_1.jpeg)

</center>

</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

例2：线图

```python

import matplotlib.pyplot as plt
import numpy as np

# 创建数据
t = np.arange(0.0, 2.0, 0.01)  # 创建一个时间数组，范围从0到2秒，步长为0.01秒
s = 1 + np.sin(2 * np.pi * t)  # 创建一个正弦波信号

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制数据
ax.plot(t, s)

# 设置坐标轴标签和标题
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')

# 添加网格线
ax.grid()

# 保存图像为文件
fig.savefig("test.png")

# 显示图形
plt.show()

```

</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

<center>

![Alt text](Figure_2.jpeg)

</center>

</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

例3：等值线 **contourf()**

```python

import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

from matplotlib import cm, ticker

N = 100
x = np.linspace(-3.0, 3.0, N)  # 创建一个横坐标数组
y = np.linspace(-2.0, 2.0, N)  # 创建一个纵坐标数组

X, Y = np.meshgrid(x, y)  # 创建坐标网格

# 创建一个低矮的圆顶和一个尖峰
# 需要在z/颜色轴上使用对数刻度，以便同时显示圆顶和尖峰
# 线性刻度只显示尖峰。
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
z = Z1 + 50 * Z2

# 在左下角放入一些负值以引发对数刻度的问题：
z[:5, :5] = -1

# 以下部分不是绝对必要的，但可以消除警告。要查看警告，请将其注释掉。
z = ma.masked_where(z <= 0, z)

# 自动选择级别可以工作；设置对数定位器告诉contourf使用对数刻度：
fig, ax = plt.subplots()
cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)

# 或者，您可以手动设置级别和规范：
# lev_exp = np.arange(np.floor(np.log10(z.min())-1),
#                    np.ceil(np.log10(z.max())+1))
# levs = np.power(10, lev_exp)
# cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())

cbar = fig.colorbar(cs)

plt.show()

```

</font>


---  



## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

<center>

![Alt text](Figure_3.jpeg)

</center>

</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

例4：子图 **subplot()**

```python

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon

# 定义一个函数f(t)，用于生成数据
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

# 创建一个时间范围
t1 = np.arange(0.0, 3.0, 0.01)

# 创建第一个子图，位置为2行1列中的第2个
ax1 = plt.subplot(212)
ax1.margins(0.05)  # 默认的边距是0.05，值为0表示自适应
ax1.plot(t1, f(t1))

# 创建第二个子图，位置为2行2列中的第1个
ax2 = plt.subplot(221)
ax2.margins(2, 2)  # 值大于0.0表示放大
ax2.plot(t1, f(t1))
ax2.set_title('Zoomed out')

# 创建第三个子图，位置为2行2列中的第2个
ax3 = plt.subplot(222)
ax3.margins(x=0, y=-0.25)  # 值在(-0.5, 0.0)之间表示缩小中心
ax3.plot(t1, f(t1))
ax3.set_title('Zoomed in')

# 显示图形
plt.show()

```

</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

<center>

![Alt text](Figure_4.jpeg)

</center>

</font>


---  



## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

例5：3D **plot_surface()**

```python

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator

# 创建一个绘图窗口和3D子图
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# 生成数据
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# 绘制三维表面
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# 自定义Z轴
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# 使用StrMethodFormatter自动设置Z轴标签格式
ax.zaxis.set_major_formatter('{x:.02f}')

# 添加颜色条，将数值映射到颜色
fig.colorbar(surf, shrink=0.5, aspect=5)

# 显示图形
plt.show()


```



</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

<center>

![Alt text](Figure_5.jpeg)

</center>

</font>


---  



## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

例6：3D **plot_surface()**

```python

import matplotlib.pyplot as plt
import numpy as np

# 创建一个绘图窗口
fig = plt.figure()

# 添加一个3D子图
ax = fig.add_subplot(projection='3d')

# 生成数据
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# 绘制三维曲面
ax.plot_surface(x, y, z)

# 设置等比例显示
ax.set_aspect('equal')

# 显示图形
plt.show()

```

</font>


---  



## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

<center>

![Alt text](Figure_6.jpeg)

</center>

</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

例7：3D **contour**

```python


import matplotlib.pyplot as plt

# 导入3D绘图工具包
from mpl_toolkits.mplot3d import axes3d

# 创建一个3D子图
ax = plt.figure().add_subplot(projection='3d')

# 获取测试数据，X、Y、Z分别代表坐标轴数据
X, Y, Z = axes3d.get_test_data(0.05)

# 绘制3D曲面
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)

# 绘制每个维度的等高线投影
ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

# 设置坐标轴范围和标签
ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='X', ylabel='Y', zlabel='Z')

# 显示图形
plt.show()

```

</font>


---  



## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

<center>

![Alt text](Figure_7.jpeg)

</center>

</font>


---  


## 2.5 △ 数据可视化（Matplotlib）
<font size=4>

contour 和 contourf 是用于绘制等高线图的两个Matplotlib函数，它们有一些联系和区别。

**联系**：

绘制等高线： 两者都用于绘制二维数据的等高线，以展示数据的轮廓和变化。

参数相似： contour 和 contourf 使用的参数大部分相似，包括数据数组、X和Y坐标、等高线级别等。

颜色映射： 两者都可以通过指定颜色映射（cmap）来调整等高线的颜色。

**区别**：

填充效果： 最主要的区别是填充效果。contour 绘制的等高线是空心的，而 contourf 绘制的等高线是实心的，可以用颜色填充等高线之间的区域。

返回值： contour 返回一个等高线集合（ContourSet），而 contourf 返回一个填充等高线集合。这意味着你可以通过不同的方法访问和处理这些对象。

视觉效果： 由于填充效果，contourf 通常在可视化上更容易理解，特别是对于展示数据分布的图。
</font>


---  








