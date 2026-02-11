#!/usr/bin/env python3
"""
分析builder.py文件的文档覆盖率
"""

import re
import sys

def analyze_file(file_path):
    """分析文件的文档覆盖率"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 查找所有方法定义
    method_pattern = r'^\s*def\s+(\w+)\s*\([^)]*\)\s*(->[^:]*)?:'
    methods = []
    
    lines = content.split('\n')
    in_example_block = False
    in_docstring = False
    
    for i, line in enumerate(lines):
        # 检查是否在示例代码块中
        if '```python' in line:
            in_example_block = True
            continue
        elif '```' in line and in_example_block:
            in_example_block = False
            continue
        
        # 跳过示例代码块中的内容
        if in_example_block:
            continue
        
        # 检查是否在文档字符串中
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            in_docstring = not in_docstring
            continue
        
        if in_docstring:
            continue
        
        # 查找方法定义
        match = re.match(method_pattern, line.strip())
        if match:
            method_name = match.group(1)
            # 检查前几行是否有文档字符串
            has_doc = False
            for j in range(max(0, i-5), i):
                if '"""' in lines[j] or "'''" in lines[j]:
                    has_doc = True
                    break
            
            # 检查方法后是否有文档字符串
            if not has_doc and i+1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line.startswith('"""') or next_line.startswith("'''"):
                    has_doc = True
            
            methods.append({
                'name': method_name,
                'line': i+1,
                'has_doc': has_doc,
                'is_private': method_name.startswith('_')
            })
    
    return methods

def print_report(methods):
    """打印分析报告"""
    total = len(methods)
    public_methods = [m for m in methods if not m['is_private']]
    private_methods = [m for m in methods if m['is_private']]
    
    documented_public = [m for m in public_methods if m['has_doc']]
    documented_private = [m for m in private_methods if m['has_doc']]
    
    print("=" * 60)
    print("文档覆盖率分析报告")
    print("=" * 60)
    print(f"总方法数: {total}")
    print(f"公共方法: {len(public_methods)}")
    print(f"私有方法: {len(private_methods)}")
    print()
    
    print("公共方法文档覆盖率:")
    print(f"  已文档化: {len(documented_public)} / {len(public_methods)}")
    if len(public_methods) > 0:
        coverage = len(documented_public) / len(public_methods) * 100
        print(f"  覆盖率: {coverage:.1f}%")
    
    print()
    print("需要文档的公共方法:")
    for method in public_methods:
        if not method['has_doc']:
            print(f"  - {method['name']} (第{method['line']}行)")
    
    print()
    print("私有方法文档覆盖率:")
    print(f"  已文档化: {len(documented_private)} / {len(private_methods)}")
    if len(private_methods) > 0:
        coverage = len(documented_private) / len(private_methods) * 100
        print(f"  覆盖率: {coverage:.1f}%")
    
    print()
    print("需要文档的私有方法 (前10个):")
    count = 0
    for method in private_methods:
        if not method['has_doc'] and count < 10:
            print(f"  - {method['name']} (第{method['line']}行)")
            count += 1
    
    if count < len([m for m in private_methods if not m['has_doc']]):
        print(f"  ... 还有{len([m for m in private_methods if not m['has_doc']]) - count}个私有方法需要文档")

def main():
    file_path = "tools/builder/base/builder.py"
    
    try:
        methods = analyze_file(file_path)
        print_report(methods)
        
        # 保存详细报告
        with open("documentation-analysis.md", "w") as f:
            f.write("# 文档分析详细报告\n\n")
            f.write(f"## 文件: {file_path}\n")
            f.write(f"分析时间: 2026-02-11\n\n")
            
            f.write("## 方法列表\n\n")
            f.write("| 方法名 | 行号 | 是否私有 | 是否有文档 |\n")
            f.write("|--------|------|----------|------------|\n")
            
            for method in methods:
                f.write(f"| {method['name']} | {method['line']} | {'是' if method['is_private'] else '否'} | {'是' if method['has_doc'] else '**否**'} |\n")
            
        print(f"\n详细报告已保存到: documentation-analysis.md")
        
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()