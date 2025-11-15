---
marp: true
theme: gaia
author: Shiyan Pan
size: 4:3
footer: '2023-09-16'
header: '第七章：Python在材料计算领域中的应用'
paginate: true
style: |
  section a {
      font-size: 100px;
  }
---



# 第七章：Python在材料计算领域中的应用
## 8.1 材料计算简介
## 8.2 △★计算方法和工具

---


## 8.1 材料计算简介

<font size=6>


材料计算是一个跨学科领域，涉及多种科学和工程领域，旨在理解、设计和优化材料的性质、结构和性能。

全球工业革命浪潮催生了材料研发的加速，各国追求低成本、高可靠性的预测方法以快速获得定制性能的新材料。

随着`大数据`时代的兴起，材料信息学领域，包括`机器学习`等人工智能技术，迅速发展为材料设计与开发的有力工具，并广泛应用于材料研究。



</font>

---




## 8.1 材料计算简介

材料科学设计原理
<font size=5>


1. 机器学习技术（包含神经网络和遗传算法）己经被证明可以有效地加速材料的研发进程。人工智能算法分为通用的`Scikit-learn`、`TensorFlow`和`Pytorch`等人工智能框架，以及针对材料性能开发的专用算法，如SISSO、AFLOW-ML、MatMiner等，这些算法有助于预测和优化材料性能。
  
![width:400](9.png) ![width:400](14.JPG)


</font>

---






## 8.1 材料计算简介

<font size=5>

2. 在材料科学研究中，建立准确的机器学习模型往往需要“海量”数据进行训练。近年来，美国、欧洲和日本等国的科研人员陆续开发了一系列高效的计算软件、材料数据库和材料人工智能算法。这些计算软件包括Materials Project、AFLOWπ、AiiDA、ASE等，用于材料计算和分析。材料数据库包括ICSD、COD、Materials Project、AFLOW-Lib、Materials Cloud、OQMD、Materials Web、NOMAD、以及日本国立材料科学研究所(NIMS)的数据库和MatNavi检索系统，这些数据库包含了大量的材料信息。主要`材料科学数据库`:

![Alt text](10.JPG)

</font>

---



## 8.1 材料计算简介

<font size=5>

以下是材料计算(`材料数据库`)涉及的主要内容：

1. `电子结构计算`：这是材料计算的核心，包括使用量子力学方法（如密度泛函理论）来模拟材料中的电子结构。这可用于预测材料的能带结构、电子密度分布和电子态密度等属性。
   
    第一性原理计算软件: VASP、Quantum Espresso、Abinit。

2. `原子和分子动力学模拟`：通过模拟原子和分子之间的相互作用以及它们在时间上的演化，可以研究材料的结构、热力学性质和响应。

    分子动力学计算软件: LAMMPS。

3. `相图计算`，CALPHAD (Computer Coupling of Phase Diagrams and Thermochemistry)：相图计算涉及研究材料在不同条件下的相变行为，例如温度、压力和成分变化。这可用于预测材料的相图、固溶度、相平衡和相变温度等。

    相图相场计算软件: OpenPhase、OpenCalphd、Thermocalc、Pandat.


</font>

---




## 8.1 材料计算简介

<font size=6>

Materials Project (MP)计算材料数据库平台(`https://www.materialsproject.org/`), 是由美国劳伦斯伯克利国家实验室(LBNL)和麻省理工学院(MIT)等单位在2011年材料基因组计划提出后联合开发的开放性数据库。

</font>

---



## 8.1 材料计算简介

<font size=6>

Materials Project数据库存储了几十万条包括能带结构、弹性张量、压电张量等性能的`第一性原理计算`数据。材料体系涉及无机化合物、纳米孔隙材料、嵌入型电极材料和转化型电极材料。

![bg right width:400](11.JPG)

</font>

---



## 8.1 材料计算简介

<font size=5>

AFLOW 计 算材料 数据库(`http://www.aflowlib.org/`),是由杜克大学在2011年开发的一个开放数据库。数据库中包含了大量`第一性原理计算`所得的数据，目前已存储了关于无机化合物、二元合金与多元合金等超过557 043 524条涉及2 945 940种材料的结构、性能数据，其中绝大多数数据都是预测得出的，是诸多数据库中数据含量最大的一个。

![bg right width:400](<12 AFLOW数据库数据量统计2020.JPG>)

</font>

---



## 8.1 材料计算简介

<font size=5>

ICSD无机晶体结构数据库(`http://icsd.fizkarlsruhe.de/`)由德国波恩大学等机构合作创建，自1913年起维护至今。它包含超过21万种晶体结构，覆盖了金属、合金、陶瓷等非有机化合物。是全球最大的无机晶体结构数据库。

![bg right width:400](<13 ICSD数据库统计2020.JPG>)

</font>

---



## 8.2 △★计算方法和工具

<font size=5>

## 8.2.1 材料结构建模和性质数据库 (PyMatGen和Matminer)
## 8.2.2 相图计算(CALPHAD, Computer Coupling of Phase Diagrams and Thermochemistry)
## 8.2.3 原子模拟环境(ASE, Atomic Simulation Environment)

</font>

---


## 8.2.1 材料结构建模和性质数据库

PyMatGen调用Materials Project数据
<font size=5>

1. `获取Materials Project API密钥`：首先，在Materials Project网站上注册并获取API密钥。这个密钥将用于身份验证和访问数据。


2. `安装所需的Python库`：需要使用Python库来与Materials Project API进行通信。一个常用的库是pymatgen，它可以帮助您处理材料数据。您可以使用以下命令安装它：

```python
pip install pymatgen
```

3. `编写Python代码`：使用Python编写代码以访问Materials Project数据。下面是一个示例代码，演示如何使用pymatgen库来获取所有单元素材料的密度数据：



</font>

---


## 8.2.1 材料结构建模和性质数据库

PyMatGen调用Materials Project数据
<font size=5>

```python
from pymatgen.ext.matproj import MPRester

# 替换成您自己的Materials Project API密钥
api_key = "YOUR_API_KEY_HERE"

# 创建MPRester对象并使用API密钥进行身份验证
with MPRester(api_key) as mpr:
    # 获取所有单元素材料的密度数据
    data = mpr.query({"elements": {"$in": ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]}}, properties=["density"])

# 打印结果
for entry in data:
    print(f"Material: {entry['material_id']}, Density: {entry['density']} g/cm^3")
```

4. `运行代码`：运行Python代码，它将使用您的API密钥与Materials Project API通信并检索所需的数据。

Materials Project API提供了许多其他可用于检索各种材料属性和数据的选项。可以根据研究需求编写不同的代码来获取所需的数据。

</font>

---


## 8.2.1 材料结构建模和性质数据库

<font size=6>

Matminer调用数据库：工作流程和功能

![width:800](10.png)

</font>

---


## 8.2.1 材料结构建模和性质数据库


<font size=6>

Matminer调用数据库：数据检索可以轻松地将复杂的在线数据放入数据框中

![width:800](11.png)
</font>

---




## 8.2.1 材料结构建模和性质数据库

<font size=5>


</font>

---




## 8.2.1 材料结构建模和性质数据库

<font size=5>


</font>

---




## 8.2.2 相图计算(CALPHAD)

**相图计算原理**
<font size=6>

等温等压条件下合金相图中热力学平衡条件依赖于化学势平衡条件.

化学势（μ）用于描述组分在不同相中的平衡。在多组分系统中，相平衡可以用化学势平衡条件表示：

$$μ_{i}^{\alpha} = μ_{i}^{\beta}$$

分别是组分$i$在相$\alpha$和相$\beta$中的化学势。

</font>

---



## 8.2.2 相图计算(CALPHAD)

<font size=5>

相平衡状态图研究由一种或数种物质所构成的相平衡系统的性质（如沸点、熔点、蒸汽压、溶解度等）与条件（如温度、压力及组成等） 的函数关系，这种关系的图叫`相图（phase diagram）`。

![width:400](9.jpg)

</font>

---


## 8.2.2 相图计算(CALPHAD)

<font size=4>

相图计算是材料科学和热力学研究中的重要任务之一，它有助于理解物质在不同温度、压力和成分条件下的相变行为。在 Python 中可以使用 pycalphad 库来进行相图计算。


| **特点** | **相图计算** | **实验测量** |
| ------- | ------------ | ----------- |
| **数据来源** | 使用热力学数据库中的数据进行计算。 | 通过实验室实际测量得到数据。 |
| **精度和可靠性** | 取决于数据库质量和模型准确性。 | 受实验条件和仪器精度的影响，有一定不确定性。 |
| **应用** | 用于预测新材料性质、相变行为。 | 用于验证已知材料性质、实验结果。 |
| **研究范围** | 可以探索广泛的组分、温度和压力范围。 | 受实验条件和材料可用性限制。 |
| **时间和成本** | 相对较快且成本较低，尤其是用于预测。 | 可能需要更多时间和资源来进行实验。 |
| **灵活性** | 可根据需要轻松修改计算条件和组分。 | 实验条件和样品准备可能受限制。 |
| **验证** | 需要与实验数据进行比较和验证。 | 通常用来验证计算结果的准确性。 |




</font>

---


## 8.2.2 相图计算(CALPHAD)

<font size=5>

pycalphad 是一个免费使用 CALPHAD 方法计算热力学的 Python 库，用于 设计热力学模型，计算相图和 用calphad方法研究相平衡。

pycalphad 提供读取Thermo Calc TDB文件和 求解多组分多相吉布斯能量最小化程序。

</font>

```python
(https://pycalphad.org/)
```

![bg right width:300](1.png)

---



## 8.2.2 相图计算(CALPHAD)

<font size=5>

1. 使用 TDB 文件计算`等压`二元相图

![width:350](6.png) ![width:350](2.png)
![width:350](3.png) ![width:350](4.png)

</font>

---


## 8.2.2 相图计算(CALPHAD)

<font size=5>

al-zn（铝锌）合合金二元相图计算

```python
# 引入 matplotlib 库并设置以内联方式显示图形
%matplotlib inline
import matplotlib.pyplot as plt

# 引入 pycalphad 库中的 Database、binplot 和 variables 模块
from pycalphad import Database, binplot
import pycalphad.variables as v

# 加载数据库文件并选择将要考虑的相位
db_alzn = Database('alzn_mey.tdb')
my_phases_alzn = ['LIQUID', 'FCC_A1', 'HCP_A3']

# 创建一个 matplotlib 的 Figure 对象并获取当前活动的 Axes 对象
fig = plt.figure(figsize=(9, 6))
axes = fig.gca()

# 计算相图并在现有的坐标轴上绘制它，使用 `plot_kwargs={'ax': axes}` 关键字参数
binplot(db_alzn, ['AL', 'ZN', 'VA'], my_phases_alzn, {v.X('ZN'): (0, 1, 0.02), v.T: (300, 1000, 10), v.P: 101325, v.N: 1}, plot_kwargs={'ax': axes})

# 显示图形
plt.show()


```

</font>


---







## 8.2.2 相图计算(CALPHAD)

<font size=5>

2. 计算二元系统的能量面
在 CALPHAD 建模中可检查系统中所有组成相的吉布斯能面。nb-re（铌铼）计算给定温度 (2800 K) 下所有相的吉布斯能量作为成分的函数。chi 相具有额外的内部自由度，使其能够针对给定的整体成分呈现多种状态。只有低能态与计算平衡相图相关。


      ![Alt text](8.png)


</font>

---


## 8.2.2 相图计算(CALPHAD)

<font size=5>

```python
# 引入matplotlib库并使用"%matplotlib inline"命令以内联方式显示图形
import matplotlib.pyplot as plt

# 引入pycalphad库中的Database、calculate、variables和plot工具中的phase_legend
from pycalphad import Database, calculate, variables as v
from pycalphad.plot.utils import phase_legend

# 引入numpy库并重命名为np
import numpy as np

# 加载热力学数据库文件'nbre_liu.tdb'，并创建数据库对象db_nbre
db_nbre = Database('nbre_liu.tdb')

# 定义要绘制的相位名称列表
my_phases_nbre = ['CHI_RENB', 'SIGMARENB', 'FCC_RENB', 'LIQUID_RENB', 'BCC_RENB', 'HCP_RENB']

# 获取相位名称到颜色的映射关系，以用于图例
legend_handles, color_dict = phase_legend(my_phases_nbre)

# 创建一个图形窗口，指定图形大小为(9, 6)，并获取坐标轴对象ax
fig = plt.figure(figsize=(9, 6))
ax = fig.gca()

# 遍历相位名称列表，计算吉布斯自由能，并散点绘制GM vs. X(RE)图形
for phase_name in my_phases_nbre:
    # 使用calculate函数计算吉布斯自由能，输出为'GM'
    result = calculate(db_nbre, ['NB', 'RE'], phase_name, P=101325, T=2800, output='GM')
    
    # 绘制散点图，X轴为'Re'的摩尔分数，Y轴为吉布斯自由能，设置散点标记、大小和颜色
    ax.scatter(result.X.sel(component='RE'), result.GM, marker='.', s=5, color=color_dict[phase_name])

# 设置图形的X轴和Y轴标签以及X轴的范围
ax.set_xlabel('X(RE)')
ax.set_ylabel('GM')
ax.set_xlim((0, 1))

# 添加图例，并设置图例位置和相对位置
ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.6))

# 显示图形
plt.show()
```

</font>

---




## 8.2.2 相图计算(CALPHAD)

<font size=5>

3. 绘制三元相图并使用三角轴
   
   在热力学中可使用二维图来表达三组分系统中的平衡。

![bg right width:500](8.png)


</font>

---




## 8.2.3 原子模拟环境(ASE)

<font size=4>

**原子模拟环境** (ASE, Atomic Simulation Environment `https://wiki.fysik.dtu.dk/ase/`) 是一个用于分子动力学和材料建模的Python库。它提供了丰富的工具和功能，使科学家和工程师能够模拟原子和分子的行为，以研究材料的性质、结构和相互作用。主要特点和功能：

1. **多样的模拟方法**：ASE支持多种分子动力学方法，包括分子动力学、Monte Carlo模拟、自洽场计算等，可以用于不同类型的材料建模。

2. **材料建模**：ASE可以用于构建材料的晶格结构、计算能带结构、分析晶格动力学，以及模拟材料的机械性能。

3. **界面友好**：ASE提供了简单的Python接口，易于使用和扩展。它还允许用户将ASE与其他科学计算库（如Numpy、SciPy）集成在一起。

4. **广泛的材料性质计算**：ASE可以用于计算材料的多种性质，包括能带结构、电子结构、热力学性质、声学性质等。

5. **可视化工具**：ASE提供了可视化工具，允许用户可视化模拟过程和结果，以便更好地理解材料的行为。

6. **兼容性**：ASE支持多种第三方软件和计算引擎，如**VASP**、**LAMMPS**、**Gaussian**等，使其能够与现有的计算工具无缝集成。

开源和活跃的社区：ASE是一个**开源项目**，拥有活跃的开发者和用户社区，不断更新和改进库的功能和性能。

</font>

---





## 8.2.3 原子模拟环境(ASE)

<font size=4>

1. 计算计算原子化能

```python
# 导入模块
from ase import Atoms  # 从ASE库导入Atoms类，用于创建原子和分子结构
from ase.calculators.emt import EMT  # 从ASE的emt模块导入EMT计算器，用于计算能量

# 创建单个氮原子
atom = Atoms('N')  # 创建一个包含单个氮原子的Atoms对象
atom.calc = EMT()  # 使用EMT计算器计算氮原子的能量
e_atom = atom.get_potential_energy()  # 获取氮原子的势能能量

# 创建氮分子
d = 1.1  # 定义氮分子的原子间距
molecule = Atoms('2N', [(0., 0., 0.), (0., 0., d)])  # 创建一个包含两个氮原子的Atoms对象
molecule.calc = EMT()  # 使用EMT计算器计算氮分子的能量
e_molecule = molecule.get_potential_energy()  # 获取氮分子的势能能量

# 计算原子化能
e_atomization = e_molecule - 2 * e_atom  # 计算原子化能，即氮分子减去两倍氮原子的能量

# 打印结果
print('氮原子能量: %5.2f eV' % e_atom)
print('氮分子能量: %5.2f eV' % e_molecule)
print('原子化能: %5.2f eV' % -e_atomization)  # 注意，这里取负号以得到正数的原子化能

输出：
Nitrogen atom energy:  5.10 eV
Nitrogen molecule energy:  0.44 eV
Atomization energy:  9.76 eV
```


</font>

---




## 8.2.3 原子模拟环境(ASE)

<font size=4>

2. 状态方程 (EOS)

首先，对不同的晶格常数进行批量计算：

```python
# 导入所需的库
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io.trajectory import Trajectory

# 定义银晶体的初基格矢和使用EMT势能计算器
a = 4.0  # 初始晶格常数
b = a / 2
ag = Atoms('Ag',
           cell=[(0, b, b), (b, 0, b), (b, b, 0)],
           pbc=1,
           calculator=EMT())

# 获取晶格参数和创建轨迹文件
cell = ag.get_cell()
traj = Trajectory('Ag.traj', 'w')

# 在不同的晶格参数下进行循环
for x in np.linspace(0.95, 1.05, 5):
    # 调整晶格参数并重新计算能量
    ag.set_cell(cell * x, scale_atoms=True)
    ag.get_potential_energy()
    # 将构型写入轨迹文件
    traj.write(ag)

```

</font>

---


## 8.2.3 原子模拟环境(ASE)

<font size=4>

编写一个轨迹文件，其中包含五种不同晶格常数的 FCC 银的五种配置。EquationOfState现在，使用类和此脚本分析结果：

```python

# 从轨迹文件中读取构型和能量数据
from ase.eos import EquationOfState
from ase.io import read
from ase.units import kJ

configs = read('Ag.traj@0:5')  # 读取5个构型
# 提取体积和能量
volumes = [ag.get_volume() for ag in configs]
energies = [ag.get_potential_energy() for ag in configs]

# 使用EquationOfState拟合状态方程
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
# 打印弹性模量（GPa）
print(B / kJ * 1.0e24, 'GPa')
# 生成状态方程图像
eos.plot('Ag-eos.png')


```

![bg right width:500](Ag-eos.png)

</font>

---



## 8.2.3 原子模拟环境(ASE)

<font size=4>

ASE（Atomic Simulation Environment）是一个功能强大的Python库，可以用来生成**VASP**（Vienna Ab Initio Simulation Package）的输入文件。ASE生成VASP输入文件的一般步骤：

1. 导入必要的ASE库：

    ```python
    from ase import Atoms
    from ase.io import write
    ```

2. 创建原子结构：

    ```python
    # 定义原子的种类和坐标
    atoms = Atoms(symbols=['H', 'O'],
                positions=[(0, 0, 0), (0, 0, 1.0)])

    # 设置晶胞参数
    atoms.set_cell([5, 5, 5])

    # 设置原子间相互作用（例如，使用Lennard-Jones势能）
    atoms.set_calculator(EMT())
    ```

3. 生成VASP输入文件：

    ```python
    # 指定输出文件名
    output_file = 'POSCAR'

    # 使用ASE的write函数生成VASP的POSCAR文件
    write(output_file, atoms, format='vasp')
    ```

生成一个名为'POSCAR'的VASP输入文件，包含原子坐标、晶格常数等信息。



</font>

---


## 8.2.3 原子模拟环境(ASE)

<font size=4>

使用ASE库中的ase.build模块创建了两个碳纳米管（Carbon Nanotube，CNT）模型，其中一个是纯碳纳米管，另一个是含有硅（Si）原子的碳纳米管。

```python
# 导入必要的模块
from ase.build import nanotube

# 创建纯碳纳米管（CNT1）
# nanotube()函数用于创建碳纳米管模型。
# 参数6表示CNT的(n, m)索引，其中n和m是整数，指定碳纳米管的结构。
# 参数0表示该碳纳米管的扩展方向是沿着Z轴。
# 参数length=4表示CNT的长度为4个单位。
cnt1 = nanotube(6, 0, length=4)

# 创建含有硅原子的碳纳米管（CNT2）
# 同样，nanotube()函数用于创建碳纳米管模型。
# 参数3和3表示CNT的(n, m)索引。
# 参数length=6表示CNT的长度为6个单位。
# 参数bond=1.4表示CNT中碳-碳键的键长为1.4 Ångströms。
# 参数symbol='Si'表示将硅（Si）原子添加到碳纳米管中。
cnt2 = nanotube(3, 3, length=6, bond=1.4, symbol='Si')
```

![Alt text](cnt1.png)  ![Alt text](cnt2.png)

</font>

---


## 8.2.3 原子模拟环境(ASE)

<font size=4>

使用ASE库中的ase.build模块创建了两种不同类型的石墨烯纳米带（Graphene Nanoribbon，GNR）模型，分别是“armchair”型和“zigzag”型，以及对这些模型进行了一些特定的设定。

```python
# 导入必要的模块
from ase.build import graphene_nanoribbon

# 创建“armchair”型石墨烯纳米带（GNR1）
# graphene_nanoribbon()函数用于创建石墨烯纳米带模型。
# 参数3和4表示GNR的(n, m)索引，这里为(3, 4)。
# 参数type='armchair'表示创建“armchair”型石墨烯纳米带。
# 参数saturated=True表示饱和边缘，即氢原子饱和。
# 参数vacuum=3.5表示在GNR上方创建3.5 Ångströms的真空层。
gnr1 = graphene_nanoribbon(3, 4, type='armchair', saturated=True, vacuum=3.5)

# 创建“zigzag”型石墨烯纳米带（GNR2）
# 同样，graphene_nanoribbon()函数用于创建石墨烯纳米带模型。
# 参数2和6表示GNR的(n, m)索引，这里为(2, 6)。
# 参数type='zigzag'表示创建“zigzag”型石墨烯纳米带。
# 参数saturated=True表示饱和边缘。
# 参数C_H=1.1和C_C=1.4表示碳-氢键的键长为1.1 Ångströms，碳-碳键的键长为1.4 Ångströms。
# 参数vacuum=3.0表示在GNR上方创建3.0 Ångströms的真空层。
# 参数magnetic=True表示在GNR上引入磁性。
# 参数initial_mag=1.12表示初始化的磁矩为1.12 Bohr magnetons。
gnr2 = graphene_nanoribbon(2, 6, type='zigzag', saturated=True, C_H=1.1, C_C=1.4,
                           vacuum=3.0, magnetic=True, initial_mag=1.12)
```

</font>

---


## 8.2.3 原子模拟环境(ASE)

<font size=4>

![Alt text](gnr1.png)  ![Alt text](gnr2.png)

</font>

---



