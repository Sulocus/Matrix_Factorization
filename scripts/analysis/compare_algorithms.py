#!/usr/bin/env python3
"""
比较 bigamp_spreading_parallel.py 和 bigamp_spreading_parallel_unit.py 的差异
验证除物理参数缩放外是否一致
"""

import difflib
import re
from pathlib import Path

def extract_functions(content: str) -> dict:
    """提取所有函数定义"""
    # 匹配函数定义
    pattern = r'^(def \w+\([^)]*\):.*?)(?=^def |\Z)'
    functions = {}
    
    lines = content.split('\n')
    current_func = None
    current_lines = []
    indent_level = 0
    
    for line in lines:
        # 检测函数定义
        if line.startswith('def '):
            if current_func:
                functions[current_func] = '\n'.join(current_lines)
            match = re.match(r'def (\w+)\(', line)
            if match:
                current_func = match.group(1)
                current_lines = [line]
                indent_level = 0
        elif current_func:
            # 检测函数结束（空行或新的顶级定义）
            if line and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('#'):
                if not line.startswith('def '):
                    functions[current_func] = '\n'.join(current_lines)
                    current_func = None
                    current_lines = []
            else:
                current_lines.append(line)
    
    if current_func:
        functions[current_func] = '\n'.join(current_lines)
    
    return functions

def normalize_code(code: str) -> str:
    """标准化代码，移除注释和空行便于比较"""
    lines = []
    for line in code.split('\n'):
        # 移除行尾注释
        line = re.sub(r'\s*#.*$', '', line)
        # 移除空行
        if line.strip():
            lines.append(line)
    return '\n'.join(lines)

def compare_files():
    base = Path('/home/sucia/Matrix_Factorization/src/smf/modules/algorithms')
    file1 = base / 'bigamp_spreading_parallel.py'
    file2 = base / 'bigamp_spreading_parallel_unit.py'
    
    content1 = file1.read_text()
    content2 = file2.read_text()
    
    print("=" * 80)
    print("文件对比：bigamp_spreading_parallel.py vs bigamp_spreading_parallel_unit.py")
    print("=" * 80)
    
    # 对比关键函数
    key_functions = [
        'bigamp_step_disjoint_union_flat',  # 核心step函数
        'compute_offset_indices',            # 索引计算
    ]
    
    # 提取函数
    funcs1 = extract_functions(content1)
    funcs2 = extract_functions(content2)
    
    print(f"\n文件1函数数量: {len(funcs1)}")
    print(f"文件2函数数量: {len(funcs2)}")
    
    # 找出共有函数和差异
    common = set(funcs1.keys()) & set(funcs2.keys())
    only_in_1 = set(funcs1.keys()) - set(funcs2.keys())
    only_in_2 = set(funcs2.keys()) - set(funcs1.keys())
    
    print(f"\n共有函数 ({len(common)}): {sorted(common)}")
    print(f"\n仅在 parallel.py ({len(only_in_1)}): {sorted(only_in_1)}")
    print(f"仅在 parallel_unit.py ({len(only_in_2)}): {sorted(only_in_2)}")
    
    # 详细对比共有的核心函数
    print("\n" + "=" * 80)
    print("核心函数详细对比")
    print("=" * 80)
    
    # 对比 bigamp_step_disjoint_union vs bigamp_step_disjoint_union_flat
    if 'bigamp_step_disjoint_union' in funcs1 and 'bigamp_step_disjoint_union_flat' in funcs2:
        print("\n### bigamp_step_disjoint_union (parallel.py) vs bigamp_step_disjoint_union_flat (unit.py)")
        
        # 提取关键公式行
        def extract_formula_lines(code):
            """提取包含关键公式的行"""
            patterns = [
                r'W_var_new\s*=',
                r'X_var_new\s*=',
                r'V\s*=.*alpha_scale',
                r'tau_W.*=',
                r'tau_X.*=',
            ]
            lines = []
            for line in code.split('\n'):
                for p in patterns:
                    if re.search(p, line):
                        lines.append(line.strip())
                        break
            return lines
        
        formulas1 = extract_formula_lines(funcs1['bigamp_step_disjoint_union'])
        formulas2 = extract_formula_lines(funcs2['bigamp_step_disjoint_union_flat'])
        
        print("\n-- parallel.py 关键公式:")
        for f in formulas1[:10]:
            print(f"   {f}")
        
        print("\n-- parallel_unit.py 关键公式:")
        for f in formulas2[:10]:
            print(f"   {f}")
    
    # 对比 compute_offset_indices
    if 'compute_offset_indices' in common:
        print("\n### compute_offset_indices 对比")
        diff = list(difflib.unified_diff(
            normalize_code(funcs1['compute_offset_indices']).split('\n'),
            normalize_code(funcs2['compute_offset_indices']).split('\n'),
            fromfile='parallel.py',
            tofile='parallel_unit.py',
            lineterm=''
        ))
        if diff:
            print("发现差异:")
            for line in diff[:20]:
                print(f"  {line}")
        else:
            print("✅ 完全一致")
    
    # 对比类的 __init__ 方法
    print("\n" + "=" * 80)
    print("类初始化方法对比（检查 torch.compile 和 BF16 设置）")
    print("=" * 80)
    
    # 搜索 __init__ 中的关键配置
    def extract_init_config(content, class_name):
        pattern = rf'class {class_name}.*?def __init__\(self.*?\):(.*?)(?=def \w+\(self)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            init_code = match.group(1)
            # 提取关键配置行
            config_lines = []
            for line in init_code.split('\n'):
                if any(kw in line for kw in ['compile', 'bf16', 'BF16', 'storage_dtype', 'use_compile']):
                    config_lines.append(line.strip())
            return config_lines
        return []
    
    config1 = extract_init_config(content1, 'BiGAMPSpreadingParallel')
    config2 = extract_init_config(content2, 'BiGAMPSpreadingParallelUnit')
    
    print("\n-- BiGAMPSpreadingParallel.__init__ 配置:")
    for c in config1:
        print(f"   {c}")
    
    print("\n-- BiGAMPSpreadingParallelUnit.__init__ 配置:")
    for c in config2:
        print(f"   {c}")
    
    # 对比 train_full_parallel 方法
    print("\n" + "=" * 80)
    print("train_full_parallel 方法对比（检查 Clone 修复）")
    print("=" * 80)
    
    def extract_train_loop(content):
        """提取训练循环的关键代码"""
        # 查找 for step in range 循环
        pattern = r'for step in range\(self\.max_steps\):(.*?)(?=return \w+_hat|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            loop_code = match.group(1)
            # 提取关键行
            key_lines = []
            for line in loop_code.split('\n'):
                if any(kw in line for kw in [
                    'cudagraph_mark_step_begin',
                    '.clone()',
                    'step_fn',
                    'CLONE STRATEGY',
                    'CRITICAL FIX'
                ]):
                    key_lines.append(line.strip())
            return key_lines
        return []
    
    loop1 = extract_train_loop(content1)
    loop2 = extract_train_loop(content2)
    
    print("\n-- parallel.py 训练循环关键代码:")
    for l in loop1:
        print(f"   {l}")
    
    print("\n-- parallel_unit.py 训练循环关键代码:")
    for l in loop2:
        print(f"   {l}")
    
    # 最终一致性检查
    print("\n" + "=" * 80)
    print("一致性总结")
    print("=" * 80)
    
    issues = []
    
    # 检查 clone 是否存在
    if '.clone()' not in content1:
        issues.append("❌ parallel.py 缺少 .clone() 调用")
    else:
        print("✅ parallel.py 包含 Clone 修复")
    
    if '.clone()' not in content2:
        issues.append("❌ parallel_unit.py 缺少 .clone() 调用")
    else:
        print("✅ parallel_unit.py 包含 Clone 修复")
    
    # 检查 torch.compile
    if 'torch.compile' not in content1:
        issues.append("❌ parallel.py 缺少 torch.compile")
    else:
        print("✅ parallel.py 包含 torch.compile")
    
    if 'torch.compile' not in content2:
        issues.append("❌ parallel_unit.py 缺少 torch.compile")
    else:
        print("✅ parallel_unit.py 包含 torch.compile")
    
    # 检查 cudagraph_mark_step_begin
    if 'cudagraph_mark_step_begin' not in content1:
        issues.append("❌ parallel.py 缺少 cudagraph_mark_step_begin")
    else:
        print("✅ parallel.py 包含 cudagraph_mark_step_begin")
    
    if 'cudagraph_mark_step_begin' not in content2:
        issues.append("❌ parallel_unit.py 缺少 cudagraph_mark_step_begin")
    else:
        print("✅ parallel_unit.py 包含 cudagraph_mark_step_begin")
    
    # 检查 BF16
    if 'storage_dtype' not in content1:
        issues.append("❌ parallel.py 缺少 BF16/storage_dtype")
    else:
        print("✅ parallel.py 包含 BF16 支持")
    
    if 'storage_dtype' not in content2:
        issues.append("❌ parallel_unit.py 缺少 BF16/storage_dtype")
    else:
        print("✅ parallel_unit.py 包含 BF16 支持")
    
    if issues:
        print("\n⚠️ 发现问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 两个文件的优化配置一致！")
    
    # 对比物理参数缩放差异
    print("\n" + "=" * 80)
    print("物理参数缩放差异（预期差异）")
    print("=" * 80)
    
    # 查找 W_var_new 和 X_var_new 的公式
    def find_variance_formula(content):
        patterns = [
            (r'W_var_new\s*=\s*[^#\n]+', 'W_var_new'),
            (r'X_var_new\s*=\s*[^#\n]+', 'X_var_new'),
        ]
        results = {}
        for pattern, name in patterns:
            matches = re.findall(pattern, content)
            if matches:
                # 去重
                results[name] = list(set(m.strip() for m in matches))
        return results
    
    formulas1 = find_variance_formula(content1)
    formulas2 = find_variance_formula(content2)
    
    print("\n-- parallel.py 方差公式:")
    for name, formulas in formulas1.items():
        for f in formulas:
            print(f"   {f}")
    
    print("\n-- parallel_unit.py 方差公式 (Unit Scaling):")
    for name, formulas in formulas2.items():
        for f in formulas:
            print(f"   {f}")
    
    print("\n预期的物理差异:")
    print("  - parallel.py:      W_var_new = 1.0 / (M + tau_W)")
    print("  - parallel_unit.py: W_var_new = 1.0 / (1.0 + tau_W)  # UNIT SCALING")

if __name__ == '__main__':
    compare_files()
