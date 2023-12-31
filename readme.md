# 赛题名称：第二届广州·琶洲算法大赛  基于多目标、多源数据扰动预测的智能排产算法(https://aistudio.baidu.com/competition/detail/950/0/introduction)
## A榜第一，B榜第二，路演第一总冠军方案
随着制造业的发展，多品种、多产线混合生产的模式已经成为主流，并在一定程度上减少制造成本，但完全根据市场需求并制定相应的产销平衡，排产排程及供应链计划，一直是制造业的难点，希望通过本活动方案，由销售端到生产端，再到供应端的优化约束排序，能实现一个合理并快速的算法及求解器。



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
# 将B榜-工厂智能排产算法赛题 .xlsx与main.py放置同一目录下
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
[初始车型路径得分, 车型切换次数得分, 
min(min([双色车间隔得分,石墨电池间隔得分, 小颜色单批内数量得分, 双色车单批内数量得分, 大颜色单批内数量得分, 石墨电池单批内数量得分]), 0.4)
天窗切换次数得分 * 4 + 颜色切换次数得分* 2  + 小颜色间隔得分 + 双色车间隔得分 + 石墨电池间隔得分 + 小颜色单批内数量得分 + 双色车单批内数量得分 + 大颜色单批内数量得分 + 石墨电池单批内数量得分， 
电池切换次数得分 + 配置等级切换次数得分 + 小颜色间隔得分]
```
其中优化第二目标的前提是第一目标不降低。
优化第三目标的前提是第一目标和第二目标得分均不降低。
优化第四目标的前提是第一目标, 第二目标和第三目标得分均不降低...
其中第3目标的设计是为了防止陷入极小值，使得部分指标得分过低，0.4的参数为个人凭调试设置
第1目标的设计是为了保证天与天之间的车型切换次数为0
* 针对切换类和单品内集中(值越小越好) 
* 得分 = (人工切换次数 – 算法切换次数) / （人工切换次数）
* 针对间隔和数量类得分(值越大越好)：
* 得分=(算法满足率 – 人工满足率) / (人工满足率)
* 
# 智能排产算法代码说明文档

本文档为所提供的智能排产算法代码的说明文档。该代码主要涉及汽车生产排产问题的解决方案，使用了基于元启发式算法（ALNS）的方法来进行排产优化。以下将对代码中的各部分进行解释和说明。

## 代码结构

代码包含了两个主要的类：`APS`（Advanced Planning and Scheduling，高级计划与排程）和`ALNS`（Adaptive Large Neighborhood Search，自适应大邻域搜索）。`APS` 类用于管理整个排产流程，而 `ALNS` 类用于执行排产优化的核心算法。

### APS 类

1. **初始化方法 (`__init__`)**: 在此方法中，`APS` 类会进行一些初始化工作，包括设置一些颜色、属性等的列表，设置批次限制和间隔限制，读取数据，并调用`prepare_data` 方法来准备数据。

2. **`update_div_dict` 方法**: 此方法用于生成一个字典，其中包含各个属性的分割值，以便在排产过程中使用。

3. **`prepare_data` 方法**: 此方法用于准备排产所需的数据。它会根据预定义的颜色和属性列表，对数据进行处理和分类，并为每个订单分配一个排产路径。

4. **`run_single` 方法**: 此方法用于运行单个排产任务，使用 `ALNS` 类进行优化。它会在一定时间内反复优化路径，直到满足终止条件。

5. **`run` 方法**: 此方法用于并行运行多个排产任务，通过多进程处理提高效率。最终，它将生成排产结果的 CSV 文件和压缩文件。

### ALNS 类

1. **初始化方法 (`__init__`)**: 在此方法中，`ALNS` 类会初始化一些参数，包括颜色、属性列表，批次限制等。它还会调用 `init_car_path` 方法来生成初始排产路径。

2. **`init_car_path` 方法**: 此方法用于生成初始的车辆排产路径，以便在后续的优化过程中使用。它会对不同日期和车型进行排列组合，以寻找可能的路径。

3. **`generate_path` 方法**: 此方法用于生成初始的排产路径，并计算初始路径的得分。它会对不同日期和车型的路径进行组织和计算得分。

4. **`objective` 方法**: 此方法用于计算路径的优化目标函数值，根据批次数量、颜色间隔等指标进行计算。

5. **`local_search` 方法**: 此方法用于执行局部搜索操作，以尝试改进当前路径。它会尝试不同的优化操作，如 2-opt、2_h_opt、relocate_move 和 exchange_move。

6. **`optimize` 方法**: 此方法用于执行排产优化的主要逻辑。它会遍历各个日期和车型，对每个路径进行局部搜索和优化。

7. **`result` 方法**: 此方法用于生成最终的排产结果，将优化后的路径映射回原始数据，并生成一个数据框来表示最终的排产方案。

## 运行方式

代码中的 `if __name__ == '__main__':` 部分会在执行脚本时调用 `APS` 类的 `run` 方法，从而开始整个排产流程。在 `run` 方法中，多个排产任务会被并行处理，最终生成排产结果的 CSV 文件和压缩文件。

## 使用的库

代码中使用了多个 Python 库，如 `os`、`pandas`、`numpy`、`itertools`、`random`、`cardinality`、`time`、`multiprocessing`、`datetime`、`zipfile` 和 `cacheout`。这些库被用于数据处理、并行处理、时间计算等方面。

## 注意事项

- 代码中的参数和常量可能需要根据实际情况进行调整，以便获得更好的排产结果。
- 代码中使用了缓存（`cacheout` 库）来存储中间结果，以减少重复计算。这可以提高代码的执行效率。



## 车型顺序初始解生成
```
获取车辆参数分组：按日期和车型将数据进行分组
初始化日期排产计划列表：为每个日期初始化一个空的排产计划列表
循环：
    对于每个日期和车型组合：
        如果车型已在排产计划中，跳过当前循环
        否则：
            尝试将车型插入不同位置，生成新的排产计划
            计算每个计划的分数
    选择最佳计划：
        从生成的计划中选择车型切换次数最少的计划
        将具有相同最低分数的计划存储在列表中
    随机选择最佳计划：
        从具有最高分数的计划列表中随机选择一个计划作为新的排产计划
    更新排产计划：
        使用新选择的排产计划更新原始排产计划
    检查循环结束条件：
        如果所有日期的车型均已加入排产计划，则结束循环
返回最终排产计划
```

## 结论

该代码实现了一种智能排产算法，用于解决汽车生产排产问题。通过并行处理和元启发式算法，它可以寻找一组优化的排产路径，以满足多个约束条件和目标函数。为了获得最佳结果，建议在实际应用中根据情况进行参数调整和优化算法的选择。

如果有任何疑问或需要进一步的帮助，请随时联系我 376030480@qq.com。
