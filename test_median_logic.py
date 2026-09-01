#!/usr/bin/env python3
"""
测试修改后的中位数逻辑
"""
import cobra
import pandas as pd
from dGbyG.api import predict_transformed_dG_prime_for_GEM

# 加载一个示例模型
print("加载代谢模型...")
gem_path = "../yeast-GEM.xml"
gem = cobra.io.read_sbml_model(gem_path)
# 或者使用 MATLAB 格式：
# gem = cobra.io.load_matlab_model('path/to/your/model.mat')

# 设置隔室条件（可选）
compartment_conditions = {
    'c': {'pH': 7.2, 'pMg': 3.0, 'I': 0.25, 'T': 298.15, 'e_potential': 0.0},
    'e': {'pH': 7.0, 'pMg': 3.0, 'I': 0.25, 'T': 298.15, 'e_potential': 0.0}
}

print("开始计算吉布斯自由能...")
Met_df, Rxn_df = predict_transformed_dG_prime_for_GEM(
    gem,
    compartment_conditions=compartment_conditions,
    use_met_id_types='all'
)

print("\n" + "="*60)
print("代谢物结果示例 (Met_df):")
print("="*60)
print(Met_df.head(10))

print("\n" + "="*60)
print("反应结果示例 (Rxn_df):")
print("="*60)
print(Rxn_df.head(10))

# 查看某个代谢物的所有标识符计算结果
print("\n" + "="*60)
print("检查某个代谢物的多个标识符结果:")
print("="*60)
met_id = Met_df.index[0]  # 取第一个代谢物
print(f"代谢物: {met_id}")
print(Met_df.loc[met_id])

# 验证中位数逻辑
print("\n" + "="*60)
print("验证中位数计算:")
print("="*60)
for rxn in gem.reactions[:3]:  # 检查前3个反应
    print(f"\n反应: {rxn.id}")
    print(f"计算的 dGr_prime: {Rxn_df.loc[rxn.id, 'dGr_prime']:.2f} kJ/mol")

    # 显示参与的代谢物及其中位数
    for met, coeff in rxn.metabolites.items():
        if met.id in Met_df.index:
            met_values = Met_df.loc[met.id].dropna()
            if len(met_values) > 0:
                # 提取均值部分（如果是元组）
                values = [v[0] if isinstance(v, tuple) else v for v in met_values]
                median_val = pd.Series(values).median()
                print(f"  {met.id} (coeff={coeff:+.1f}): 中位数 = {median_val:.2f} kJ/mol, 共{len(values)}个标识符")

print("\n测试完成！")
