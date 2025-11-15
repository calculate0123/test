---
marp: true
theme: gaia
author: Shiyan Pan
size: 4:3
footer: '2023-09-16'
header: '第五章：Python在人工智能领域的应用 - 机器学习'
paginate: true
style: |
  section a {
      font-size: 100px;
  }
---



# 第五章：Python在人工智能领域的应用

<font size=5>

## 5.1 人工智能、机器学习和深度学习简介
## 5.2 机器学习模型构建和常用库

</font> 


---

## 5.1 人工智能、机器学习和深度学习简介

<font size=5>

- AI的定义

  - 人工智能（Artificial Intelligence，AI）是一门计算机科学领域，旨在创建具备智能的计算机系统，使其能够模仿人类的智能行为。这包括学习、推理、问题解决、感知、语言理解和决策制定等各种智能任务。AI的目标是使计算机系统能够在特定领域或任务中表现出类似人类的智能水平，甚至在某些情况下超越人类智能。

  - AI系统通过使用大量数据和复杂的算法来模拟和实现各种智能任务。AI技术包括机器学习、深度学习、自然语言处理、计算机视觉、强化学习等。这些技术允许计算机系统从经验中学习，不断改进性能，并适应新的情境和问题。

</font> 

---


## 5.1 人工智能、机器学习和深度学习简介

<font size=3>

- AI的发展趋势

| 发展趋势                              | 描述                                                                                       |
|-------------------------------------|--------------------------------------------------------------------------------------------|
| 深度学习的持续发展                   | 深度学习技术的不断改进和扩展，包括更复杂的神经网络架构和更大规模的数据集。            |
| 自监督学习                           | 自监督学习方法的兴起，减少了对大规模标记数据的依赖。                                   |
| 增强学习的应用扩展                   | 增强学习在自动驾驶、机器人控制、供应链管理等领域的应用扩展。                           |
| 医疗保健领域的增长                   | AI在医学图像分析、疾病诊断和药物研发方面的不断增长。                                  |
| 自然语言处理（NLP）的发展           | NLP技术的进步，能够更好地理解和生成自然语言。                                         |
| AI与边缘计算的结合                   | AI与边缘计算相结合，用于实现实时决策和智能边缘设备。                                 |
| 伦理和法律问题                       | 伦理和法律问题在AI应用中引起越来越多的关注和监管。                                   |
| AI的教育和培训                     | 培养AI专业人才的教育和培训资源的增加，填补技能鸿沟。                                  |
| AI在可持续发展中的作用               | AI技术在能源管理、环境监测和可持续发展领域的应用。                                   |



</font> 

---


## 5.1 人工智能、机器学习和深度学习简介
- AI的应用领域
 
<font size=3>

| 应用领域                      | 描述                                                                                             |
|------------------------------|--------------------------------------------------------------------------------------------------|
| 自动驾驶汽车                 | AI用于实现自动驾驶汽车，包括感知、决策和控制，以提高交通安全和效率。                         |
| 医疗保健                     | AI用于医学影像分析、疾病诊断、药物研发和健康监测，以改善医疗保健的效率和准确性。             |
| 金融服务                     | AI应用于风险管理、股票交易、信用评估和客户服务，以提高金融决策的效率和精确性。             |
| 自然语言处理（NLP）           | NLP用于文本分析、语音识别、机器翻译和虚拟助手，以改善与计算机的自然交互。                 |
| 电子商务                     | 推荐系统和个性化推荐利用AI来提高在线购物的体验，促进销售增长。                             |
| 工业自动化                   | AI在制造业中应用，包括机器人控制、质量检测和供应链优化，提高生产效率。                     |


</font> 

---

## 5.1 人工智能、机器学习和深度学习简介
- AI的应用领域

<font size=3>


| 应用领域                      | 描述                                                                                             |
|------------------------------|--------------------------------------------------------------------------------------------------|
| 农业                         | 农业领域使用AI进行作物监测、智能灌溉和预测农产品产量，以提高农业生产效率。               |
| 游戏                         | AI用于游戏中的虚拟敌人、智能决策和游戏设计，提供更具挑战性和沉浸感的游戏体验。             |
| 教育                         | 教育领域应用AI以个性化教育、在线学习和智能教育工具，提高学习效果。                         |
| 航空航天                     | AI用于飞行控制、自主飞行和卫星导航，提高航空航天系统的安全性和效率。                     |
| 环境监测                     | AI在环境监测中用于气象预测、空气质量检测和自然灾害预警，保护环境和人们的安全。           |
| 物联网（IoT）                | AI与物联网设备结合，实现智能家居、智慧城市和智能工厂等应用，提高生活和工作效率。         |
| 媒体和娱乐                   | AI在内容推荐、视频分析和虚拟角色创造中用于媒体和娱乐产业，提供个性化体验。               |


</font> 

---





## 5.1 人工智能、机器学习和深度学习简介

<font size=4>

| 领域         | 描述                                              | 联系                                   | 区别                                       |
| ------------ | ------------------------------------------------- | -------------------------------------- | ------------------------------------------ |
| 机器学习     | 一种人工智能方法，使计算机从数据中学习和做出预测。 | 机器学习是人工智能的一个子领域。     | 机器学习是广义的概念，包括各种学习算法，不限于神经网络。    |
| 深度学习     | 机器学习的子领域，使用深度神经网络解决复杂问题。   | 深度学习是机器学习的一种方法，依赖于神经网络。 | 深度学习特指使用多层神经网络进行学习，更适用于处理大规模和复杂数据。 |
| 人工智能     | 让计算机模仿人类智能行为以解决问题的领域。        | 机器学习和深度学习是实现人工智能的工具。  | 人工智能是更广泛的概念，包括各种智能方法和技术，不限于机器学习和深度学习。  |
| 神经网络     | 由多个层次的神经元组成，用于模拟人脑处理信息的方式。 | 深度学习依赖于神经网络，是神经网络的一种应用。  | 神经网络是一种特定的模型，用于实现深度学习，但不限于此。    |


</font> 

---


## 5.1 人工智能、机器学习和深度学习简介

<font size=4>


![width:700](8.1.jpg)

</font> 

---


## 5.1 人工智能、机器学习和深度学习简介

<font size=4>


| 伦理和社会影响               | 描述                                                                                   |
|-----------------------------|----------------------------------------------------------------------------------------|
| 隐私问题                    | AI对大量个人数据的访问引发隐私问题，需要强化数据保护和隐私法规。                   |
| 歧视和公平性                | AI算法可能反映数据偏见，导致对某些群体的不公平对待，需要确保公平和消除歧视。     |
| 解释性和透明性              | 复杂的AI系统通常难以解释其决策过程，需要更多的透明性和解释性。                     |
| 就业市场变革                | AI的自动化可能影响传统工作岗位，需要重新思考技能培训和就业政策。                 |
| 创新和竞争                  | AI在企业和国际竞争中的作用不断增强，需要投资于AI研发以保持竞争力。               |
| 安全性和恶意用途            | AI系统面临网络攻击、数据泄露和恶意用途的威胁，需要强化安全性措施。             |
| 健康和医疗领域              | AI在医疗保健中的应用有潜力，但需要解决隐私、伦理和责任问题以确保安全性。         |
| 社会不平等                  | AI技术普及不均可能加剧社会不平等，需要关注包容性和公平性。                       |

</font> 

---



## 5.1 人工智能、机器学习和深度学习简介

<font size=5>

`机器学习`步骤：

1. 数据收集与准备：收集与问题相关的数据，清理和准备数据以确保数据质量。这包括数据清洗、特征选择和标签分配。

2. 特征工程：在这一步骤中，数据特征被选择、转换和创建，以使其适用于机器学习算法。这有助于提高模型性能。

3. 模型选择：选择适当的机器学习算法，根据问题类型（分类、回归等）和数据集特点来确定最佳算法。

4. 模型训练：使用训练数据集来训练选定的模型，使其学会模式并进行预测。


</font>

---




## 5.1 人工智能、机器学习和深度学习简介

<font size=5>

`机器学习`步骤：

5. 模型评估：通过使用验证数据集来评估模型的性能，以了解模型的准确性和泛化能力。

6. 超参数调整：调整模型的超参数，以进一步改善模型性能。

7. 模型部署：一旦满意的模型被创建，它可以部署到生产环境中，用于实际应用。


</font>

---




## 5.1 人工智能、机器学习和深度学习简介

<font size=5>

`深度学习`步骤：

1. 数据准备：与机器学习类似，深度学习也需要数据准备，但更注重大规模、高维度的数据。

2. 构建`神经网络`：在深度学习中，关键步骤是构建深层神经网络，包括定义网络结构、层次和激活函数的选择。

3. 初始化参数：初始化神经网络的权重和偏置参数，通常使用随机初始化。

4. `前向传播`：将数据通过神经网络传递，从输入层到输出层，计算模型的预测值。

5. `损失函数`：定义损失函数，用于测量模型的性能，通常是预测值与实际值之间的误差。


</font>

---



## 5.1 人工智能、机器学习和深度学习简介

<font size=5>

`深度学习`步骤：


6. `反向传播`：通过反向传播算法，根据损失函数的梯度，更新神经网络的参数，以最小化损失。

7. 训练和优化：重复前向传播和反向传播过程，直到模型收敛，即损失函数足够小。

8. 模型评估：与机器学习一样，深度学习模型需要在验证数据集上进行评估。

9. 超参数调整：调整神经网络的超参数，如学习率、批量大小，以优化性能。

10. 模型部署：最终模型可部署用于实际应用。


</font>

---




## 5.1 人工智能、机器学习和深度学习简介

<font size=2.5>

`机器学习`和`深度学习`有的联系与区别

| 特征                  | 机器学习(scikit-learn)               | 深度学习(pytorch、TensorFlow/keras)               |
|-----------------------|------------------------|------------------------|
| 基本概念              | 一种广义的学习方法，通过算法使计算机从数据中学习并提高性能。 | 一种机器学习的子领域，侧重于使用`神经网络`模型进行学习。         |
| 数据需求              | 通常需要手工提取和选择特征。 | 通常无需手工提取特征，可以从原始数据中学习特征表示。      |
| 特征工程              | 特征工程通常是手动的过程，需要领域知识。 | 在深度学习中，特征工程的需求较少，模型可以自动提取特征。   |
| 算法选择              | 通常使用各种传统算法，如决策树、支持向量机、随机森林等。 | 深度学习主要侧重于神经网络，如卷积神经网络（CNN）和循环神经网络（RNN）。 |
| 计算资源需求          | 相对较低，通常可以在常规硬件上运行。 | 对计算资源的需求较高，通常需要GPU或TPU等加速硬件。       |
| 大数据和高维度数据    | 机器学习可以处理大数据和高维度数据，但需要谨慎选择算法。 | 深度学习在处理大数据和高维度数据时具有显著的优势。        |
| 解释性和可解释性      | 通常较容易解释模型的预测结果。 | 深度学习模型通常更难以解释，被认为是黑盒模型。           |
| 适用领域              | 广泛应用于图像处理、自然语言处理、推荐系统等领域。 | 主要应用于图像识别、语音识别、自然语言处理等领域。         |
| 知名应用              | 随机森林、XGBoost、SVM等。  | 卷积神经网络（CNN）、循环神经网络（RNN）、Transformer等。 |

</font>

---



## 5.1 人工智能、机器学习和深度学习简介

<font size=5>

**监督学习**和**无监督学习**是机器学习的两种主要类型

- **监督学习**：

  - 监督学习是一种机器学习方法，其中**使用标记数据**训练模型。
  - 在监督学习中，模型需要找到映射函数来映射输入变量$X$和输出变量$Y$。
  - 监督学习需要监督来训练模型，这类似于学生在老师面前学习东西。
  - 监督学习可用于两类问题：**分类**和**回归**。


</font>

---


## 5.1 人工智能、机器学习和深度学习简介

<font size=5>

- **无监督学习**：

  - 无监督学习也称为无监督机器学习，它使用机器学习算法来分析**未标记的数据集**并进行聚类。
  - 这些算法无需人工干预，即可发现隐藏的模式或数据分组。
  - 这种方法能够发现信息的相似性和差异性，因而是探索性数据分析、交叉销售策略、客户细分和图像识别的理想解决方案。
  - 常见的无监督学习方法无监督学习模型用于执行三大任务：**聚类**、**关联**和**降维**。

</font>

---




## 5.2 机器学习模型构建和常用库

<font size=5>

- sklearn
  - sklearn，全称scikit-learn，是python中的机器学习库，建立在numpy、scipy、matplotlib等数据科学包的基础之上，支持机器学习中包括分类，回归，降维和聚类四大机器学习算法。还包括了特征提取，数据处理和模型评估三大模块。
  - 与深度学习库存在pytorch、TensorFlow等多种框架可选不同，sklearn是python中传统机器学习的首选库。

``` 
https://scikit-learn.org  
``` 

</font>

---





## 5.2 机器学习模型构建和常用库
- scikit-learn的基本功能
![width:600](6/6.1.png)
---



<!-- 常用的回归：线性、决策树、SVM、KNN ；集成回归：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees
常用的分类：线性、决策树、SVM、KNN，朴素贝叶斯；集成分类：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees
常用聚类：k均值（K-means）、层次聚类（Hierarchical clustering）、DBSCAN
常用降维：LinearDiscriminantAnalysis、PCA
　　这个流程图代表：蓝色圆圈是判断条件，绿色方框是可以选择的算法，我们可以根据自己的数据特征和任务目标去找一条自己的操作路线。
　　sklearn中包含众多数据预处理和特征工程相关的模块，虽然刚接触sklearn时，大家都会为其中包含的各种算法的广度深度所震惊，但其实sklearn六大板块中有两块都是关于数据预处理和特征工程的，两个板块互相交互，为建模之前的全部工程打下基础。 -->


## 5.2 机器学习模型构建和常用库
- scikit-learn的基本功能
  - 数据预处理 Preprocessing
  - 数据降维 Dimensionality reduction
  - 模型选择 Model selection
  - 分类 Classification
  - 回归 Regression
  - 聚类 Clustering

---

## 5.2 机器学习模型构建和常用库
<font size=4>

- 数据预处理 `Preprocessing`
  - 数据预处理，是指数据的**特征提取**和**归一化**，是机器学习过程中的第一个也是最重要的一个环节。
  - **归一化**是指将输入数据转换为具有零均值和单位权方差的新变量，但因为大多数时候都做不到精确等于零，因此会设置一个可接受的范围，一般都要求落在0-1之间。
  - **特征提取**是指将文本或图像数据转换为可用于机器学习的数字变量。
  - 需要特别注意的是，这里的特征提取与数据降维中提到的特征选择非常不同。特征选择是指通过去除不变、协变或其他统计上不重要的特征量来改进机器学习的一种方法。


<center>

![width:400](6/6.7.png)

</center>

</font>


---


## 5.2 机器学习模型构建和常用库
<font size=4>

- 数据降维 `Dimensionality reduction`
  - 数据降维，是指使用主成分分析（PCA）、非负矩阵分解（NMF）或特征选择等降维技术来**减少要考虑的随机变量的个数**，其主要应用场景包括可视化处理和效率提升。

</font>


<center>

![width:500](6/6.5.png)

</center>

---

## 5.2 机器学习模型构建和常用库
<font size=4>

- 模型选择 `Model selection`
  - 模型选择，是指对于**给定参数和模型的比较、验证和选择**，其主要目的是通过参数调整来提升精度。目前scikit-learn实现的模块包括：格点搜索，交叉验证和各种针对预测误差评估的度量函数。 

</font>

<center>

![width:500](6/6.6.png)

</center>


---



## 5.2 机器学习模型构建和常用库
<font size=4>

- 分类 `Classification`
  - 分类算法，是指识别给定对象的所属类别，属于**监督学习**的范畴，最常见的应用场景包括垃圾邮件检测和图像识别等。
  - 目前scikit-learn已经实现的算法包括：支持向量机（SVM），最近邻，逻辑回归，随机森林，决策树以及多层感知器（MLP）神经网络等等。
  - 需要指出的是，由于scikit-learn本身不支持深度学习，也不支持GPU加速，因此这里对于MLP的实现并不适合于处理大规模问题。
 
</font>

![width:850](6/6.2.png)

---


## 5.2 机器学习模型构建和常用库
<font size=4>

- 回归 `Regression`
  - 回归算法，是指预测与给定对象相关联的连续值属性，最常见的应用场景包括预测药物反应和预测股票价格的**无监督学习**的方法。
  - 目前scikit-learn已经实现的算法包括：支持向量回归（SVR），脊回归，Lasso回归，弹性网络（Elastic Net），最小角回归（LARS ），贝叶斯回归，以及各种不同的鲁棒回归算法等。 

</font>


<center>

![width:400](6/6.3.png)

</center>

---

## 5.2 机器学习模型构建和常用库
<font size=4>

- 聚类 `Clustering`
  - 聚类算法，是指自动识别具有相似属性的给定对象，并将其分组为集合，属于**无监督学习**的范畴，最常见的应用场景包括顾客细分和试验结果分组。
  - 目前scikit-learn已经实现的算法包括：K-均值聚类，谱聚类，均值偏移，分层聚类，DBSCAN聚类等。 

</font>


<center>

![width:400](6/6.4.png)

</center>


---



## 5.2 机器学习模型构建和常用库

- scikit-learn多元回归分析

<font size=5>

scikit-learn的传统线性回归(最小二乘法)：

`linear_model.LinearRegression`

以一个简单的房屋价格预测作为例子来解释线性回归的基本要素。

目标是预测一栋房子的售出价格（元）。

这个价格取决于很多因素，如房屋状况、地段、市场行情等。

为了简单起见，假设价格只取决于房屋状况的两个因素，即**面积**（平方米）和**房龄**（年）。

接下来我们希望探索价格与这两个因素的具体关系。




</font>



---


## 5.2 机器学习模型构建和常用库


<font size=5>


![bg right width:400](linear.JPG)


设房屋的面积为 $x_1$，房龄为 $x_2$，售出价格为 $y$。我们需要建立基于输入 $x_1$ 和 $x_2$ 来计算输出 $y$ 的表达式，也就是模型（model）。顾名思义，线性回归假设输出与各个输入之间是线性关系：

<center>

$\hat{y}=x_{1} w_{1}+x_{2} w_{2}+b$

</center>


其中 $\hat{y}$ 是预测值。

<!-- <div style="text-align:center">

$\hat{y}=x_{1} w_{1}+x_{2} w_{2}+b$

</div> -->



</font>



---



## 5.2 机器学习模型构建和常用库

- scikit-learn多元回归分析

<font size=5>

<div style="text-align:center">

$\hat{y}=x_{1} w_{1}+x_{2} w_{2}+b$

</div>


其中 $w_1$ 和 $w_2$ 是权重（weight），$b$ 是偏差（bias），且均为标量。它们是线性回归模型的参数（parameter）。$\hat{y}$是预测值，$x_1$和$x_2$是两个特征，$w_1$和$w_2$是相应的权重（系数），$b$是截距项。

目标变量（因变量）：$y_i$（$i$代表数据点的索引）
特征变量（自变量）：$x_{1i}$和$x_{2i}$（两个`特征`，$i$代表数据点的索引）
预测模型：$\hat{y_i} = x_{1i}w_1 + x_{2i}w_2 + b$
残差项（误差项）：$e_i = y_i - \hat{y}_i$
假设残差项$e_i$服从正态分布：$e_i \sim \mathcal{N}(0, \sigma^2)$



</font>

---



## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

<font size=4.5>


使用最小二乘法找到系数$w_1$、$w_2$和截距项$b$，使残差平方和最小化。损失函数定义为残差平方和：

$$S = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

为了找到系数的解析解，我们可以对$w_1$、$w_2$和$b$的偏导数分别设置为零。

首先，对$w_1$求偏导数：

$$\frac{\partial S}{\partial w_1} = \sum_{i=1}^{n} -2x_{1i}(y_i - \hat{y}_i)$$

$$= -2 \sum_{i=1}^{n} x_{1i}(y_i - (x_{1i}w_1 + x_{2i}w_2 + b)) = 0$$


</font>

---


## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

<font size=4>

对$w_2$求偏导数：

$$\frac{\partial S}{\partial w_2} = \sum_{i=1}^{n} -2x_{2i}(y_i - \hat{y}_i)$$

$$= -2 \sum_{i=1}^{n} x_{2i}(y_i - (x_{1i}w_1 + x_{2i}w_2 + b)) = 0$$

对 $b$ 求偏导数：

$$\frac{\partial S}{\partial b} = \sum_{i=1}^{n} -2(y_i - \hat{y}_i)$$

$$= -2 \sum_{i=1}^{n} (y_i - (x_{1i}w_1 + x_{2i}w_2 + b)) = 0$$

</font>

---



<!-- 
## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

<font size=5>

解这个线性方程组将得到$w_1$、$w_2$和$b$的解析解，它们是多元线性回归模型的系数。最终的`解析解`为：

$$w_1 = \frac{\sum_{i=1}^{n} x_{1i}(\hat{y}_i - y_i)}{\sum_{i=1}^{n} x_{1i}^2}$$

$$w_2 = \frac{\sum_{i=1}^{n} x_{2i}(\hat{y}_i - y_i)}{\sum_{i=1}^{n} x_{2i}^2}$$

$$b = \frac{\sum_{i=1}^{n} (\hat{y}_i - y_i)}{n}$$


</font>

--- -->



## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

<font size=5>

**矩阵形式**：需要将关于$w_{1}=\beta_{1}$, $w_{2}=\beta_{2}$, ..., $w_{p}=\beta_{p}$, $b=\beta_{0}$的方程表示为矩阵形式，然后使用线性代数的方法求解比较方便：

$y=X\beta+\epsilon$ 

$$
\mathrm{Y}=\left[\begin{array}{c}
\mathrm{y}_{1} \\
\mathrm{y}_{2} \\
\vdots \\
\mathrm{y}_{\mathrm{n}}
\end{array}\right], \mathrm{X}=\left[\begin{array}{cccc}
1 & \mathrm{x}_{11} & \cdots & \mathrm{x}_{1 \mathrm{p}} \\
1 & \mathrm{x}_{21} & \cdots & \mathrm{x}_{2 \mathrm{p}} \\
\vdots & \vdots & \vdots & \vdots \\
1 & \mathrm{x}_{\mathrm{n} 1} & \cdots & \mathrm{x}_{\mathrm{np}}
\end{array}\right], \boldsymbol{\beta}=\left[\begin{array}{c}
\beta_{0} \\
\beta_{1} \\
\vdots \\
\beta_{\mathrm{p}}
\end{array}\right], \boldsymbol{\varepsilon}=\left[\begin{array}{c}
\varepsilon_{1} \\
\varepsilon_{2} \\
\vdots \\
\varepsilon_{\mathrm{n}}
\end{array}\right],
$$

其中：$Y$ 是一个 $n×1$ 列向量，表示因变量。$X$ 是一个 $n×p$ 矩阵，表示自变量的设计矩阵。$β$ 是一个 $p×1$ 列向量，表示回归系数。$ϵ$ 是一个 $n×1$ 列向量，表示误差项。$ϵ∼N(0, \sigma^2I)$ 表示误差项服从均值为零、方差为 $σ^2$  的多元正态分布。特例：系数$w_1 = β_1$、$w_2 = β_2$和截距项$b = β_0$

</font>

---


## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

<font size=5>

确定一个 $\hat{\boldsymbol{\beta}}$ 使得 $\boldsymbol{\varepsilon}=Y-X\boldsymbol{\beta}$ 各元素的平方和达到最小：

$$
\begin{aligned}
\mathrm{Q}(\boldsymbol{\beta}) & =\sum_{\mathrm{i}=1}^{\mathrm{n}} \varepsilon_{\mathrm{i}}^{2} \\
& =\boldsymbol{\varepsilon}^{\mathrm{T}} \boldsymbol{\varepsilon} \\
& =(\mathrm{Y}-\mathrm{X} \boldsymbol{\beta})^{\mathrm{T}}(\mathrm{Y}-\mathrm{X} \boldsymbol{\beta}) \\
& =\left(\mathrm{Y}^{\mathrm{T}} \mathrm{Y}-2 \boldsymbol{\beta}^{\mathrm{T}} \mathrm{X}^{\mathrm{T}} \mathrm{Y}+\boldsymbol{\beta}^{\mathrm{T}} \mathrm{X}^{\mathrm{T}} \mathrm{X} \boldsymbol{\beta}\right)
\end{aligned}

$$

</font>

---




## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

<font size=5>

对$\boldsymbol{\beta}$求导（$\frac{\partial \mathrm{x}^{\mathrm{T}} \mathrm{a}}{\partial \mathrm{x}}=\frac{\partial \mathrm{a}^{\mathrm{T}} \mathrm{x}}{\partial \mathrm{x}}=\mathrm{a}$, $\frac{\partial \mathrm{x}^{\mathrm{T}} \mathrm{Ax}}{\partial \mathrm{x}}=\mathrm{Ax}+\mathrm{A}^{\mathrm{T}} \mathrm{x}$）：
$$
\frac{\partial \mathrm{Q}(\boldsymbol{\beta})}{\partial \beta}=-2 \mathrm{X}^{\mathrm{T}} \mathrm{Y}+2 \mathrm{X}^{\mathrm{T}} \mathrm{X} \boldsymbol{\beta}=0
$$

得到：
$$
\mathrm{X}^{\mathrm{T}} \mathrm{X} \boldsymbol{\beta}=\mathrm{X}^{\mathrm{T}} \mathrm{Y}
$$


</font>

---





## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

<font size=5>


$$
\mathrm{X}^{\mathrm{T}} \mathrm{X} \boldsymbol{\beta}=\mathrm{X}^{\mathrm{T}} \mathrm{Y}
$$


由于 $\operatorname{rank}\left(\mathrm{X}^{\mathrm{T}} \mathrm{X}\right)=\operatorname{rank}(\mathrm{X})=\mathrm{p}+1$，则得到 $X^T X$ 是正定矩阵，X $X^T X$ 存在逆矩阵， 就有唯一解了：

$$
\hat{\boldsymbol{\beta}}=\left(\hat{\boldsymbol{\beta}}_{0}, \hat{\boldsymbol{\beta}}_{1}, \cdots, \hat{\boldsymbol{\beta}}_{\mathrm{p}}\right)^{\mathrm{T}}=\left(\mathrm{X}^{\mathrm{T}} \mathrm{X}\right)^{-1} \mathrm{X}^{\mathrm{T}} \mathrm{Y}
$$




此时$\boldsymbol{\beta}$的估计就得到了，如果再把它带回到模型中去就有：

$$
\hat{Y}=X \hat{\boldsymbol{\beta}}=\mathrm{X}\left(\mathrm{X}^{\mathrm{T}} \mathrm{X}\right)^{-1} \mathrm{X}^{\mathrm{T}} \mathrm{Y}=\mathrm{SY}
$$


</font>

---





## 5.2 机器学习模型构建和常用库
- scikit-learn逻辑回归分析(分类线性模型)

<font size=4>

Logistic回归（Logistic Regression，简称LR）是一种常用的处理二类分类问题的模型。在二类分类问题中，把因变量y可能属于的两个类分别称为负类和正类，则因变量y∈{0, 1}，其中0表示负类，1表示正类。

<center>

![Alt text](logistic-1.png)

</center>

</font>

---



## 5.2 机器学习模型构建和常用库

<font size=4>

- scikit-learn逻辑回归分析(分类线性模型)

逻辑回归作为分类线性模型实现的，而不是按照 scikit-learn/ML 命名法的回归模型。逻辑回归在文献中也称为 logit 回归、最大熵分类 (MaxEnt) 或对数线性分类器。

在此模型中，使用**逻辑函数**$f(x)$对描述单个试验的可能结果的概率进行建模。**逻辑函数**$f(x)$ 是一种sigmoid函数，表达式为：

$$f(x) = \frac{1}{1 + e^{-x}}$$

预测值：

$$\hat{y}=f(b + \sum_{i} w_i x_i)$$ 
$$ln \frac{\hat{y}}{1-\hat{y}} = b + \sum_{i} w_i x_i$$



</font>

---



## 5.2 机器学习模型构建和常用库

<font size=5>


**逻辑回归的误差函数**，也被称为损失函数，是由极大似然估计推导出来的。对于二分类问题，我们可以定义损失函数如下

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1-y^{(i)}) \log(1-\hat{p}^{(i)})]$$

其中，$m$ 是样本数量，$y^{​(i)}$ 是第 i 个样本的真实标签，$p^{​(i)}$ 是模型对第 $i$ 个样本的预测概率。这个损失函数的目标是最小化模型预测概率与真实标签之间的差异。

- **凸函数**：逻辑回归的误差函数是处处可微的凸函数，其曲线的谷底对应最小的误差。这意味着我们可以使用梯度下降等优化算法找到其全局最小值。
- **连续且可微**：这使得我们可以使用基于梯度的优化算法（如梯度下降）来最小化损失函数。
- **非负**：损失函数的值总是大于或等于零，当且仅当模型的预测概率完全等于真实标签时，损失函数的值为零。

</font>

---





## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析

| 符号标记  | 描述          |
|---------|--------------|
| X_train | 训练数据(训练).    |
| X_test  | 测试数据(评估).    |
| X       | 完整数据.    |
| y_train | 训练集标签(训练).  |
| y_test  | 测试集标签(评估).  |
| y       | 数据标签.    |


---


## 5.2 机器学习模型构建和常用库
- scikit-learn多元回归分析
  - 导入工具包：
```python
# 导入scikit-learn库中的datasets模块和preprocessing模块
from sklearn import datasets, preprocessing

# 导入scikit-learn库中的train_test_split函数，用于分割数据集
from sklearn.model_selection import train_test_split

# 导入scikit-learn库中的LinearRegression线性回归模型
from sklearn.linear_model import LinearRegression

# 导入scikit-learn库中的r2_score函数，用于计算R²分数来评估模型性能
from sklearn.metrics import r2_score
```

---


## 5.2 机器学习模型构建和常用库
<font size=4>

- scikit-learn多元回归分析
  - 加载数据：
    - Scikit-learn支持以NumPy的arrays对象、Pandas对象、SciPy的稀疏矩阵及其他可转换为数值型arrays的数据结构作为其输入，前提是数据必须是数值型的。
    - sklearn.datasets模块提供了一系列加载和获取著名数据集如鸢尾花、波士顿房价、Olivetti人脸、MNIST数据集等的工具，也包括了一些toy data如S型数据等的生成工具。

```python
# 导入scikit-learn库中的load_iris函数，用于加载鸢尾花数据集
from sklearn.datasets import load_iris

# 使用load_iris函数加载鸢尾花数据集，将数据集存储在变量iris中
iris = load_iris()

# 从iris变量中提取特征数据（鸢尾花的测量数据），并将其存储在变量X中
X = iris.data

# 从iris变量中提取目标数据（鸢尾花的类别标签），并将其存储在变量y中
y = iris.target
```
</font>

---



## 5.2 机器学习模型构建和常用库
<font size=4>

load_iris数据集是机器学习领域中经常用于示例和练习的经典数据集之一。这个数据集包含了鸢尾花的特征和类别信息，用于分类问题。 load_iris数据集的详细描述：

`数据来源`：这个数据集最早由统计学家和生物学家Ronald A. Fisher在1936年收集。它包含了来自三种不同鸢尾花品种（Setosa、Versicolor和Virginica）的样本数据。

`特征`：每个样本包含四个特征，这些特征是鸢尾花的四个形态特征，包括花萼（sepal）的长度和宽度，以及花瓣（petal）的长度和宽度，都以厘米为单位。因此，每个样本有四个特征。

`目标变量`：除了特征数据之外，每个样本还有一个对应的目标标签，表示鸢尾花的品种。共有三个类别，分别代表三种不同的鸢尾花。

`总样本数`：load_iris数据集包含150个样本，其中每种鸢尾花品种各有50个样本。

```python
#view data description and information
print(iris.DESCR)


import pandas as pd
#make sure to save the data frame to a variable
data = pd.DataFrame(iris.data)
data.head()

#note: it is common practice to use underscores between words, and avoid spaces
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

data.head()
```

</font>

---



## 5.2 机器学习模型构建和常用库
<font size=6>

  - 数据划分
```python
#  将完整数据集的70%作为训练集，30%作为测试集
#  并使得测试集和训练集中各类别数据的比例与原始数据集比例一致（stratify分层策略）
#  另外可通过设置 shuffle=True 提前打乱数据

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test =  
train_test_split(X, y, random_state=12, stratify=y,  test_size=0.3)
```
</font>

---

## 5.2 机器学习模型构建和常用库
<font size=6>

  - 数据预处理

```python
# 导入scikit-learn库中的StandardScaler类，用于特征标准化
from sklearn.preprocessing import StandardScaler

# 创建一个StandardScaler对象，该对象将用于对数据进行特征标准化
scaler = StandardScaler()

# 使用fit_transform方法，对训练数据集 X_train 进行特征标准化的拟合和转换操作
# 特征标准化的目的是将特征数据进行缩放，使其具有零均值和单位方差
# 这有助于提高模型性能，尤其是对于某些机器学习算法，如支持向量机和K均值聚类
# X_train是训练数据集中的特征数据，经过拟合和转换后，标准化后的特征数据将存储在新的变量中
# 通常，这个新的变量不仅会存储标准化后的数据，还会覆盖原始的X_train
scaler.fit_transform(X_train)
```

$$x^* = \frac{x - \mu}{\sigma}，\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)})^2，\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$$


</font>

---

## 5.2 机器学习模型构建和常用库

<font size=5>

  - 数据预处理

| 预处理器/变换器   | 描述                                       |
|-------------------|--------------------------------------------|
| `MinMaxScaler`     | 将特征缩放到指定范围内，通常是 [0, 1]。   |
| OneHotEncoder     | 将分类特征转换为二进制形式。              |
| Normalizer        | 对每个样本的特征进行标准化或归一化。       |
| Binarizer         | 将数值特征二进制化，根据阈值进行转换。    |
| LabelEncoder      | 将类别标签转换为整数标签。                |
| Imputer           | 用于处理缺失数据，替换缺失值为统计值。    |
| PolynomialFeatures| 生成原始特征的多项式特征。                |


`MinMaxScaler`： 
$$x^* = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$


</font>

---

## 5.2 机器学习模型构建和常用库

<font size=5>

  - 特征选择

| 特征选择方法                   | 描述                                                         | 主要参数                                                      |
|-------------------------------|--------------------------------------------------------------|---------------------------------------------------------------|
| `SelectKBest(score_func, k)`  | 选择K个最重要的特征。                                      | `score_func`: 评分函数，`k`: 特征数量。                        |
| `RFECV(estimator, scoring)`   | 递归特征消除交叉验证，通过逐步删除不重要的特征。         | `estimator`: 模型估计器，`scoring`: 评分标准（默认为 "r2"）。 |
| `SelectFromModel(estimator)`  | 根据模型的特征重要性选择最重要的特征。                   | `estimator`: 带有特征重要性评估功能的模型估计器。           |


</font>


---



## 5.2 机器学习模型构建和常用库

<font size=5>

  - 监督学习算法-回归

```python
# 导入Scikit-Learn库中的线性回归模型，即 LinearRegression
from sklearn.linear_model import LinearRegression
# 构建LinearRegression 类的实例，命名为 lr。normalize=True 表示在拟合模型时对特征进行标准化。
lr = LinearRegression(normalize=True)
# 训练模型，使用 fit 方法拟合线性回归模型
lr.fit(X_train, y_train)
# 作出预测，使用 predict 方法对测试数据集 X_test 进行预测，将结果存储在 y_pred 变量中。
y_pred = lr.predict(X_test) 

LASSO	linear_model.Lasso # 基于L1正则化的线性回归模型。
Ridge	linear_model.Ridge #  是基于L2正则化的线性回归模型。
ElasticNet	linear_model.ElasticNet # 同时使用L1和L2正则化的线性回归模型。
回归树 tree.DecisionTreeRegressor # 一种非线性回归模型，通过构建树结构来建模数据的关系。

```

</font>

---

## 5.2 机器学习模型构建和常用库

<font size=5>

  - 监督学习算法-分类

```python
# 导入决策树分类器类
from sklearn.tree import DecisionTreeClassifier

# 创建一个决策树分类器的实例
clf = DecisionTreeClassifier(max_depth=5)
# 这里的 max_depth 是树的最大深度，可以根据问题调整以控制树的复杂性

# 使用训练数据拟合（训练）分类器
clf.fit(X_train, y_train)
# X_train 是训练数据集的特征，y_train 是相应的目标值

# 使用训练好的分类器进行预测
y_pred = clf.predict(X_test)
# X_test 是测试数据集的特征，y_pred 存储了预测的分类标签

# 获取分类的概率估计值
y_prob = clf.predict_proba(X_test)
# X_test 是测试数据集的特征，y_prob 存储了每个类别的概率估计值
# 使用决策树分类算法解决二分类问题， y_prob 为每个样本预测为“0”和“1”类的概率

```
</font>


---

## 5.2 机器学习模型构建和常用库

<font size=5>

  - 监督学习算法-分类

| 分类器/方法                        | 描述                                                         |
|-----------------------------------|--------------------------------------------------------------|
| `linear_model.LogisticRegression` | 逻辑回归模型，用于分类任务。                                 |
| `svm.SVC`                         | 支持向量机分类器，用于数据分割成类别。                       |
| `naive_bayes.GaussianNB`          | 朴素贝叶斯分类器，适用于处理连续型特征的分类问题。           |
| `neighbors.NearestNeighbors`      | 最近邻搜索方法，通常用于无监督学习和数据降维。               |


</font>


---


## 5.2 机器学习模型构建和常用库

<font size=5>

  - 评价指标

    sklearn.metrics模块包含了一系列用于评价模型的评分函数、损失函数以及成对数据的距离度量函数. from sklearn.metrics import accuracy_score  accuracy_score(y_true, y_pred). 对于测试集而言，y_test即是y_true，大部分函数都必须包含真实值y_true和预测值y_pred.

    - **回归模型评价**
  metrics.mean_absolute_error() | 平均绝对误差MAE
  metrics.mean_squared_error() | 均方误差MSE
  metrics.r2_score() | 决定系数R2.

    - **分类模型评价**
  metrics.accuracy_score() | 正确率 metrics.precision_score() | 各类精确率 metrics.f1_score() | F1 值   
  metrics.log_loss() | 对数损失或交叉熵损失 metrics.confusion_matrix | 混淆矩阵
  metrics.classification_report | 含多种评价的分类报告

</font>


---


## 5.2 机器学习模型构建和常用库
<font size=5>

使用 AdaBoost 的决策树回归
1. 准备具有正弦关系和一些高斯噪声的虚拟数据: 

```python

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

import numpy as np

rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

```

</font>

---


## 5.2 机器学习模型构建和常用库
<font size=5>

使用 AdaBoost 的决策树回归:
2. 使用决策树和 AdaBoost 回归器进行训练和预测

```python
# 现在，我们定义分类器并将它们拟合到数据中。 
# 然后，我们根据相同的数据进行预测，看看它们能很好地拟合它。
# 第一个回归量是 with. 第二个回归器是一个具有 of 作为基本学习器，
# 并将使用这些基本学习器构建。
# DecisionTreeRegressormax_depth
# =4AdaBoostRegressorDecisionTreeRegressormax_depth=4n_estimators=300
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)
```

</font>

---


## 5.2 机器学习模型构建和常用库
<font size=5>

使用 AdaBoost 的决策树回归
3. 拟合数据: 

```python

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("colorblind")

plt.figure()
plt.scatter(X, y, color=colors[0], label="training samples")
plt.plot(X, y_1, color=colors[1], label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, color=colors[2], label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()

```

</font>

---


## 5.2 机器学习模型构建和常用库
<font size=5>

![width:700](6/6.3.png)

</font>

---




## 5.2 机器学习模型构建和常用库  scikit-learn支持向量机

<font size=5>

- scikit-learn支持向量机


</font>

---