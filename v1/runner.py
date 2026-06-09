# runner.py (V2 - 增加了数据预检功能)
# 这个脚本是一个用户友好的启动器，用于运行 dea_calculator.py。
# 它会询问您所有必要的参数，然后自动执行计算，无需您手动修改任何代码。

import sys
import os
import re
import subprocess
import pandas as pd
import numpy as np

# --- 常量定义 ---
CALCULATOR_SCRIPT = "dea_calculator.py"
DATA_FILE = "t1.xlsx"
OUTPUT_FILE = "allindex.xlsx"
TEMP_SCRIPT = "temp_dea_calculator.py"

def get_user_input(prompt, input_type=int):
    """一个健壮的函数，用于从用户那里获取指定类型的输入。"""
    while True:
        try:
            user_input = input(prompt)
            return input_type(user_input)
        except ValueError:
            print(f"输入无效！请输入一个有效的整数。")
        except Exception as e:
            print(f"发生错误: {e}")
            return None

def validate_data(params):
    """在运行前检查数据和参数的有效性。"""
    print("\n" + "-"*60)
    print("🔬 正在执行数据预检...")
    
    try:
        df = pd.read_excel(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ 预检失败: 找不到数据文件 '{DATA_FILE}'。")
        return False
    except Exception as e:
        print(f"❌ 预检失败: 读取Excel文件时出错: {e}")
        return False

    is_valid = True
    
    # 检查1: DMU 和 Period 的一致性
    dmus = params['dmus']
    periods = params['periods']
    total_rows = len(df)
    
    if total_rows != dmus * periods:
        print(f"❌ 预检失败: 数据行数与参数不匹配！")
        print(f"   - 您输入的 DMU 数量: {dmus}")
        print(f"   - 您输入的时期数量: {periods}")
        print(f"   - 预期总行数应为: {dmus} * {periods} = {dmus * periods}")
        print(f"   - 但 t1.xlsx 文件实际总行数为: {total_rows}")
        print("   - 👉 请确认每个时期都有且仅有 {dmus} 条数据。")
        is_valid = False

    # 检查2: 投入产出变量列是否存在
    nx = params['nx']
    ny = params['ny']
    nb = params['nb']
    undesirable = params['undesirable']
    
    expected_cols = 2 + nx + ny + (nb if undesirable == 1 else 0)
    if df.shape[1] < expected_cols:
        print(f"❌ 预检失败: 投入/产出变量数量与数据文件的列数不符。")
        print(f"   - 根据您的参数，数据至少应有 {expected_cols} 列。")
        print(f"   - 但 t1.xlsx 文件只有 {df.shape[1]} 列。")
        is_valid = False

    # 检查3: 投入产出数据中是否存在0或负数
    input_cols_end = 2 + nx
    output_cols_end = input_cols_end + ny + (nb if undesirable == 1 else 0)
    
    try:
        data_values = df.iloc[:, 2:output_cols_end].values.astype(float)
        if np.any(data_values <= 0):
            print("⚠️  预检警告: 在投入或产出列中检测到零值或负值。")
            print("   - 这很可能会导致计算错误 (ValueError: c must not contain inf)。")
            print("   - 👉 建议用一个极小的正数（如0.0001）替换所有零值或负值。")
            
            proceed = input("   - 是否要忽略此警告并继续？ (y/n): ").lower()
            if proceed != 'y':
                is_valid = False
    except (ValueError, IndexError):
         print("⚠️  预检警告: 无法将投入/产出列转换为数值进行检查。请确保这些列都是数字。")


    if is_valid:
        print("✅ 预检通过！数据和参数看起来是匹配的。")
    print("-" * 60)
    return is_valid


def interpret_results():
    """对生成的 allindex.xlsx 文件进行简单的解读。"""
    # ... (此函数无需改动，代码与之前版本相同) ...
    if not os.path.exists(OUTPUT_FILE):
        print(f"\n错误：未找到结果文件 '{OUTPUT_FILE}'。无法进行解读。")
        return

    print("\n" + "="*60)
    print("            🚀 结果快速解读 🚀")
    print("="*60)

    try:
        # 读取最关键的两个指数表
        df_ml = pd.read_excel(OUTPUT_FILE, sheet_name='FGLR1992_C')
        df_gm = pd.read_excel(OUTPUT_FILE, sheet_name='PL2005_C')

        # --- 1. 总体生产率解读 (Malmquist Index) ---
        avg_ml = df_ml['mlc(M指数)'].mean()
        avg_ecc = df_ml['ecc(效率变化)'].mean()
        avg_tcc = df_ml['tcc(技术进步)'].mean()

        print("\n--- 1. Malmquist (ML) 生产率指数分析 (基于FGLR1992) ---\n")
        print(f"所有决策单元在所有时期内的平均ML指数为: {avg_ml:.4f}")
        if avg_ml > 1:
            print(f"解读: 总体来看，生产率平均每个时期增长了 {(avg_ml - 1):.2% }。")
        elif avg_ml < 1:
            print(f" 解读: 总体来看，生产率平均每个时期下降了 {(1 - avg_ml):.2% }。")
        else:
            print(" 解读: 总体来看，生产率没有显著变化。")
        
        print("\n驱动因素分解:")
        print(f"  - 平均效率变化 (追赶效应): {avg_ecc:.4f} ({'提升' if avg_ecc > 1 else '下降'})")
        print(f"  - 平均技术进步 (前沿移动): {avg_tcc:.4f} ({'进步' if avg_tcc > 1 else '退步'})")
        print("\n 这意味着生产率的变化主要是由 " + ("'自身效率提升'" if abs(avg_ecc-1) > abs(avg_tcc-1) else "'整体技术进步'") + " 驱动的。")

        # --- 2. 全局生产率解读 (Global Malmquist Index) ---
        avg_gm = df_gm['mgc(GM指数)'].mean()
        avg_bpcc = df_gm['bpcc(技术差距变动)'].mean()

        print("\n\n--- 2. 全局 Malmquist (GM) 生产率指数分析 (基于PL2005) ---\n")
        print(f"所有决策单元在所有时期内的平均GM指数为: {avg_gm:.4f}")
        if avg_gm > 1:
            print(f" 解读: 基于全局技术前沿，生产率平均每个时期增长了 {(avg_gm - 1):.2% созна}。")
        else:
            print(f" 解读: 基于全局技术前沿，生产率平均每个时期下降了 {(1 - avg_gm):.2% созна}。")
        
        print("\n驱动因素分解:")
        print(f"  - 平均技术差距变化 (BPC): {avg_bpcc:.4f}")
        print("BPC > 1 意味着当期的技术前沿正在向'历史最佳'技术前沿靠拢（技术在追赶）。")

        # --- 3. 识别优秀与待改进的决策单元 ---
        dmu_performance = df_ml.groupby('dmu')['mlc(M指数)'].mean().sort_values(ascending=False)
        print("\n\n--- 3. 决策单元 (DMU) 表现排名 (基于平均ML指数) ---\n")
        print("表现最佳的前5名 DMU:")
        print(dmu_performance.head(5).to_string())
        print("\n表现有待改进的后5名 DMU:")
        print(dmu_performance.tail(5).to_string())
        
    except Exception as e:
        print(f"\n解读结果时发生错误: {e}")
        print("请检查 'allindex.xlsx' 文件是否存在且格式正确。")

    print("\n" + "="*60)
    print("解读完毕。详细数据请查看 allindex.xlsx 文件的各个工作表。")
    print("="*60)


def main():
    """主函数，引导用户完成整个流程。"""
    print("="*60)
    print("        欢迎使用 SBM-Malmquist 效率分析工具 (V2)")
    print("="*60)

    # --- 1. 检查必要文件 ---
    if not os.path.exists(CALCULATOR_SCRIPT):
        print(f"错误：计算脚本 '{CALCULATOR_SCRIPT}' 不存在！")
        print("请确保您已将 1.txt 重命名为 dea_calculator.py 并放在同一目录下。")
        input("按 Enter 键退出。")
        return
    if not os.path.exists(DATA_FILE):
        print(f"错误：数据文件 '{DATA_FILE}' 不存在！")
        print("请确保您的Excel数据文件已命名为 t1.xlsx 并放在同一目录下。")
        input("按 Enter 键退出。")
        return

    print("\n请根据您的数据文件 (t1.xlsx) 回答以下问题：\n")

    # --- 2. 从用户获取参数 ---
    params = {
        "dmus": get_user_input("➡️  请输入决策单元的数量 (DMUs): "),
        "periods": get_user_input("➡️  请输入观察时期的数量 (Periods): "),
        "nx": get_user_input("➡️  请输入投入变量的数量 (Inputs): "),
        "ny": get_user_input("➡️  请输入期望产出变量的数量 (Good Outputs): "),
        "nb": get_user_input("➡️  请输入非期望产出变量的数量 (Bad Outputs): "),
        "undesirable": get_user_input("➡️  是否存在非期望产出？ (1代表是, 0代表否): "),
        "sup": get_user_input("➡️  是否需要计算超效率？ (1代表是, 0代表否): ")
    }
    
    # --- 新增步骤: 执行数据预检 ---
    if not validate_data(params):
        print("\n由于预检未通过，程序已停止。请修正问题后重试。")
        input("按 Enter 键退出。")
        return
        
    print("\n" + "-"*60)
    print("参数确认：")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    print("-"*60)
    
    confirm = input("\n参数是否正确？ (y/n): ").lower()
    if confirm != 'y':
        print("\n操作已取消。")
        input("按 Enter 键退出。")
        return

    # --- 3. 动态修改并运行脚本 ---
    print("\n 正在准备运行计算脚本，请稍候...")

    try:
        with open(CALCULATOR_SCRIPT, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        modified_code = original_code
        for key, value in params.items():
            pattern = re.compile(rf"^{key}\s*=\s*\d+", re.MULTILINE)
            replacement = f"{key} = {value}"
            modified_code = pattern.sub(replacement, modified_code)
            
        with open(TEMP_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(modified_code)

        print("开始执行计算... 这可能需要几分钟时间，请耐心等待。")
        print("下面的进度条由 'dea_calculator.py' 脚本生成：\n")

        subprocess.run([sys.executable, TEMP_SCRIPT], check=True)

        print("\n\n计算完成！结果已保存到 'allindex.xlsx' 文件中。")

    except FileNotFoundError:
        print(f"错误：无法读取脚本 '{CALCULATOR_SCRIPT}'。")
    except subprocess.CalledProcessError:
        print("\n 在执行计算脚本时发生错误。在预检通过后仍然出错，")
        print("   这可能意味着您的投入/产出数据列中存在非数值（文本）内容。请再次检查数据。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
    finally:
        if os.path.exists(TEMP_SCRIPT):
            os.remove(TEMP_SCRIPT)
            print("已清理临时文件。")

    # --- 4. 询问是否进行结果解读 ---
    if os.path.exists(OUTPUT_FILE):
        interpret_choice = input("\n是否需要对结果进行简单的自动解读？ (y/n): ").lower()
        if interpret_choice == 'y':
            interpret_results()

    print("\n感谢使用！")
    input("按 Enter 键退出。")

if __name__ == "__main__":
    main()