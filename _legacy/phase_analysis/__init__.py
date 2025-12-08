"""
真·Phase Analyzer - 精密相变分析框架

核心原则:
1. Data-driven 而非 assumption-driven
2. 分步实现，分步验证
3. LLM 视觉仅用于中间过程验证，最终结果依赖数学分析

模块结构:
- core/: Mode 1 基础分析 (梯度计算、曲线分类、相变检测、异常检测、区域分类)
- training/: BiG-AMP 训练封装
- adaptive/: Mode 2 自适应采样
- precise/: Mode 3 精密分析 (热力学极限、dense limit)
- validation/: 验证模块
- visualization/: 可视化模块
"""

__version__ = "0.1.0"
