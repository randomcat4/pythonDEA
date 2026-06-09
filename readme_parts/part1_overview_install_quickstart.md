# SBM-Malmquist DEA 效率分析工具

## 项目概述

本项目是为学术研究、论文写作设计的 **SBM-Malmquist DEA 效率分析工具**，专为实现基于SBM模型的DEA效率评估与Malmquist生产率指数计算而开发。

### 核心功能

- 支持基于SBM（Slack-Based Measure）模型的DEA效率分析
- 提供多种Malmquist生产率指数分解方法：FGLR1992、FGLR1994、RD1997、Zofio2007、PL2005
- 同时支持 **CRS**（常规模报酬）与 **VRS**（可变规模报酬）两种假设
- 支持 **非期望产出**（Bad Outputs）与 **超效率计算**（Super Efficiency）

### 开发目的

为提升研究效率，该工具旨在简化DEA分析流程，使用户能够在无需编程经验的前提下，快速完成复杂的效率评估任务。

### 主要特性

- ✅ 支持非期望产出（Bad Outputs）
- ✅ 支持超效率计算（Super Efficiency）
- ✅ 支持多种Malmquist指数分解方法
- ✅ 同时计算CRS和VRS两种规模报酬假设下的效率值
- ✅ 友好的交互式界面，无需修改任何代码即可运行

### 技术栈

- **Python 3.7+**
- 依赖库：`pandas`、`numpy`、`scipy`、`tqdm`、`openpyxl`

---

## 安装和环境配置指南

### Python 版本要求

本工具要求 Python 3.7 或更高版本。推荐使用 Python 3.9 或更新版本以获得最佳兼容性。

### 安装依赖

执行以下命令以安装所需依赖库：

```bash
pip install pandas numpy scipy tqdm openpyxl
```

### 环境验证

安装完成后，可使用以下代码验证环境是否配置成功：

```python
import pandas as pd
import numpy as np
import scipy
import tqdm
import openpyxl
print("依赖库安装成功！")
```

### 常见问题解决方案

1. **pip 下载慢**：可切换为国内镜像源（如清华源）安装：
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy scipy tqdm openpyxl
   ```
2. **权限问题**（Windows）：尝试以管理员身份运行命令行

---

## 快速开始 / 傻瓜式操作指南

### 准备数据文件

1. 准备一个名为 `t1.xlsx` 的Excel文件，包含所有决策单元（DMUs）的投入产出数据
2. 数据应按"时期 → 决策单元"的顺序排列。例如，4个时期、5个DMU，则共有20行数据
3. 数据格式示例如下：

| DMU | 年份 | 投入1 | 投入2 | 产出1 | 产出2 | 非期望产出 |
|-----|------|-------|-------|-------|-------|------------|
| A   | 2000 | 10    | 20    | 15    | 25    | 3          |
| B   | 2000 | 12    | 22    | 16    | 28    | 4          |
| A   | 2001 | 11    | 19    | 17    | 27    | 2          |
| B   | 2001 | 13    | 21    | 18    | 30    | 3          |

### 运行方式

#### 方式一：双击启动脚本

双击项目目录下的 `新建文本文档.bat`，工具将自动启动并进行交互式参数输入。

#### 方式二：通过命令行执行

```bash
python runner.py
```

### 交互式参数输入流程

运行后会提示您输入如下参数：

```
➡️  请输入决策单元的数量 (DMUs): 30
➡️  请输入观察时期的数量 (Periods): 5
➡️  请输入投入变量的数量 (Inputs): 3
➡️  请输入期望产出变量的数量 (Good Outputs): 4
➡️  请输入非期望产出变量的数量 (Bad Outputs): 1
➡️  是否存在非期望产出？ (1代表是, 0代表否): 1
➡️  是否需要计算超效率？ (1代表是, 0代表否): 1
```

### 预期等待时间

根据数据规模不同，计算时间约为 **1-10分钟**。请耐心等待进度条完成。

### 结果文件位置

计算完成后，结果将保存在当前目录下的 `allindex.xlsx` 文件中。该文件包含多个工作表，每个对应一种Malmquist指数的计算结果。运行完成后可选择是否进行自动解读。
