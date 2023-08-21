## 算法运行环境
* centos/ubuntu
* Anaconda3-2023.07-2-Linux-x86_64
## 预期运行时间
24小时
## 代码运行方法
```
# 下载最新版本anconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
# 安装最新版本anconda
sh Anaconda3-2023.07-2-Linux-x86_64.sh
# 将B榜-工厂智能排产算法赛题 .xlsx与135.87386.py放置同一目录下
nohup python main.py &
tail -f nohup.out 
# 24h后会生成结果至out文件夹 并以时间戳命名 该文件夹下包含commit.zip文件
```
## 模型构建思路及调优过程
### 算法目标
1.	切换（例如车型切换，前后两辆车车型编号不同，车型切换+1，切换指标希望越小越好）
2.	间隔（间隔指的是两批有要求的属性的车的间隔数量，例如要求小颜色A的车和小颜色B的车两批之间有60台及以上非小颜色车，如果间隔大于等于60，满足约束，小于就满足，针对这类约束会统计它的满足率）
3.	单批内数量（例如要求小颜色A一批数量大于等于15小于等于30，数量满足这批就满足约束，单批内的衡量也算满足率）

### 目标函数构造
由于车型切换为绝对优先，电池切换，配置等级切换和小颜色间隔优先级为0，故目标函数构造为
```
[车型切换次数得分, 
min(min([小颜色间隔得分, 双色车间隔得分,石墨电池间隔得分, 小颜色单批内数量得分, 双色车单批内数量得分, 大颜色单批内数量得分, 石墨电池单批内数量得分]), 0.4)
天窗切换次数得分 * 4 + 颜色切换次数得分* 2  + 小颜色间隔得分 + 双色车间隔得分 + 石墨电池间隔得分 + 小颜色单批内数量得分 + 双色车单批内数量得分 + 大颜色单批内数量得分 + 石墨电池单批内数量得分， 
电池切换次数得分 + 配置等级切换次数得分]
```
其中优化第二目标的前提是第一目标不降低。
优化第三目标的前提是第一目标和第二目标得分均不降低。
优化第四目标的前提是第一目标, 第二目标和第三目标得分均不降低。
其中第二目标的设计是为了防止陷入极小值，使得部分指标得分过低，0.4的参数为个人凭调试设置
* 针对切换类和单品内集中(值越小越好) 
* 得分 = (人工切换次数 – 算法切换次数) / （人工切换次数）
* 针对间隔和数量类得分(值越大越好)：
* 得分=(算法满足率 – 人工满足率) / (人工满足率)
* 
# 代码说明文档

本文档旨在解释所提供的代码段的功能、结构和使用方法。代码是一个用于工厂智能排产算法竞赛的Python脚本，其主要目的是实现工厂生产订单的优化排程。

## 功能概述

该代码通过多个函数和类来实现对工厂生产订单的排产和优化，具体包括以下功能：

1. **数据预处理：** 从Excel文件读取原始数据，并根据预定义的颜色和属性进行分类处理，生成用于排产的数据集。

2. **路径生成与优化：** 基于预处理后的数据，生成车辆排产路径，并通过局部搜索算法对路径进行优化，以达到排产的最优化目标。

3. **目标函数计算：** 计算排产方案的目标函数值，包括车辆切换次数、批次间隔、批次数量等指标。

4. **结果输出：** 将优化后的排产方案保存为CSV文件，并打包为ZIP文件。

## 代码结构

代码由以下几部分组成：

1. **导入模块和定义常量：** 导入必要的Python模块，定义一些常量和列表用于后续处理。

2. **函数定义：**
   - `update_div_dict()`: 更新划分字典，用于分割订单数量。
   - `flatten(nested_list)`: 将嵌套列表展平为一维列表。
   - `prepare_data()`: 数据预处理，对原始数据进行清洗和分类处理，生成排产所需的数据集。

3. **APS类定义：** 排产类，包含以下主要方法：
   - `__init__(self, df)`: 初始化方法，接受预处理后的数据集，并初始化属性。
   - `init_car_path(self)`: 初始化车辆路径，生成初始车辆排产路径。
   - `generate_path(self)`: 生成车辆路径并进行优化，获取排产方案的路径和得分。
   - ...（省略其他方法说明）

4. **主函数和多进程运行：** `main(plan_id)`函数对数据进行预处理、排产路径生成和优化，并返回优化结果；`run()`函数调用主函数进行多进程并行运行，最终将优化结果保存为CSV文件和ZIP文件。

## 使用方法

1. **准备数据：** 将原始数据文件（Excel格式）命名为 "B榜-工厂智能排产算法赛题 .xlsx"，与代码文件放置在同一目录下。

2. **运行代码：** 在Python环境中运行代码文件，确保相关的Python库已安装（例如：numpy、pandas、multiprocessing等）。

3. **结果输出：** 运行结束后，将生成一个名为 "commit.csv" 的CSV文件，其中包含了经过优化的排产方案。

4. **结果打包：** 代码还会将优化结果打包为一个ZIP文件，命名为 "commit.zip"，包含了生成的CSV文件。

## 注意事项

- 代码中的参数和限制条件（如车型切换次数、批次数量限制等）可能需要根据实际情况进行调整。

- 代码中使用了多进程进行优化，可根据实际计算机性能进行适当的调整。

- 运行时间较长，可能需要一段时间才能得到优化结果。

- 代码使用的数据和环境需要与代码一致，确保文件路径和数据结构正确。

- 为了保证运行稳定，可以根据需要对代码进行代码调试和优化。

## 总结

该代码段实现了工厂生产订单的排产优化，通过多个函数和类实现了数据预处理、路径生成、优化等功能。用户可根据实际情况对代码进行调整和优化，以满足特定的排产需求。
