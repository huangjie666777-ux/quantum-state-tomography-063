# Quantum Feedback SDK

一个可嵌入的小型量子电路模拟 SDK，使用密度矩阵确定性处理测量、经典反馈、reset 和相位翻转噪声，不使用随机轨迹近似。0.2 版本新增变分态制备：RZ 角度可绑定命名参数，针对 Pauli 能量目标（支持经典后选）计算确定性期望、精确梯度并进行局部最小化训练。

## 安装与接口

开发环境使用：

```bash
.venv/bin/python -m pip wheel --no-build-isolation --no-deps . -w dist
```

主入口是 `run(num_qubits, num_clbits, operations)`，其中量子位数和经典位数都必须是 1 到 6 的整数，所有量子位和经典位初始为 0。`operations` 为按时间顺序执行的映射列表：

- `{"op": "H" | "X" | "Z", "qubit": q}`：单量子位门。
- `{"op": "RZ", "qubit": q, "angle": theta}`：角度单位为弧度。
- `{"op": "CX", "control": c, "target": t}`：控制位与目标位不能相同。
- `{"op": "MEASURE", "qubit": q, "clbit": c}`：计算基测量并覆写经典位。
- `{"op": "RESET", "qubit": q}`：将该量子位非破坏性地重置为 0，保留其他量子位的约化状态，不修改经典位。
- `{"op": "PHASE_FLIP", "qubit": q, "p": p}`：确定性混合信道 `(1-p)ρ + p ZρZ`，`0 <= p <= 1`。
- 任意门可加 `"condition": [c, v]`：只有经典位 `c` 等于 `v`（0 或 1）时执行。

返回的 `ExecutionResult` 提供：

- `density_matrix`：末态量子密度矩阵副本。
- `classical_probabilities`：经典位串到概率的映射。
- `samples(shots, seed)`：使用局部 `numpy.random.Generator` 采样，不污染全局 NumPy 随机状态。
- `sample_counts(shots, seed)`：返回位串计数。

## 位序

量子位 0 对应计算基整数索引的最低位。例如 2 量子位 Bell 态是 `|00> + |11>`，其基态索引为 0 和 3。经典输出位串高位在左：经典位 0 的值显示在最右侧，未测量经典位保持 0，因此 3 个经典位中只有 `c0=1` 时输出 `001`。

测量后，模拟器保留带概率的归一化条件密度矩阵分支。后续条件门只作用于匹配的经典分支；不同读数分支不发生相干干涉。只有经典记录完全相同的分支才按概率混合。经典位被覆写后，相同新读数的分支也会按概率混合。

## 变分模板与参数绑定

RZ 角度除常数外还可以是命名参数的缩放加偏移：`scale * param + offset`。

- 简写：`{"op": "RZ", "qubit": q, "angle": "theta"}`。
- 完整形式：`{"op": "RZ", "qubit": q, "angle": {"param": "theta", "scale": -2.0, "offset": 0.1}}`。
- `scale` 默认为 1，`offset` 默认为 0；`scale` 必须是非零有限数，因此允许负缩放但不允许退化为常数的零缩放。
- 同一参数可以出现在任意多个门中（共享参数），每个门都可以有自己的缩放和偏移。

典型流程是：

```python
template = validate_template(num_qubits, num_clbits, operations)
bound_operations = bind_parameters(template, {"theta": 0.7})
result = execute(bound_operations, num_qubits, num_clbits)
```

`validate_parameters(template, mapping)` 先完整校验映射：参数必须齐全、不能有多余键、值必须是有限数值。绑定生成全新的常量操作列表，模板和调用方传入的映射都不会被修改。常数电路仍直接使用 `run`；向 `run` 传入参数化角度会抛出 `CircuitValidationError`。测量、反馈条件、reset 和 PHASE_FLIP 噪声在绑定后原样保留。

## 能量目标与后选

`energy(num_qubits, num_clbits, operations, hamiltonian, parameter_values=None, postselection=None)` 确定性计算能量期望，不抽样。哈密顿量是实系数 Pauli 串之和，每项写作 `(coeff, "IXYZ...")` 或 `{"coeff": coeff, "pauli": "IXYZ..."}`：

- 每个 Pauli 串长度必须恰好等于量子位数，字符只允许 `I`、`X`、`Y`、`Z`。
- 字符串右端对应量子位 0，左端对应最高量子位；例如 2 量子位的 `"IZ"` 表示 Z 作用在量子位 0。
- 系数必须有限。非法目标抛出 `HamiltonianValidationError`。

`postselection` 是经典位到要求取值（0 或 1）的映射，例如 `{0: 0}` 表示只保留末尾经典寄存器中 c0=0 的分支；未指定时使用全部分支。返回的 `EnergyResult` 包含 `energy`（后选时即条件能量）、`conditional_energy`、`success_probability` 和所用的 `postselection`。成功概率不超过 `1e-12` 时抛出 `PostselectionError`，不会返回伪造的条件能量。

## 精确梯度

`gradient(num_qubits, num_clbits, operations, hamiltonian, parameter_values, postselection=None)` 返回每个命名参数的精确梯度字典。实现对每个参数化 RZ 门独立使用 π/2 参数移位规则，然后按各自身的缩放（含负数）求和，因此共享参数、负缩放和条件门都被正确处理。条件能量是比值 N/P，分子 N（选中分支的 Pauli 期望加权和）与归一化概率 P 分别移位求导，后选能量的归一化项也被精确计入。该方法不是有限差分，也不依赖抽样。移位后的电路即使某些后选分支概率为零，也只按未归一化量组合，不会让基点有效的梯度失败。

## 确定性局部训练

`minimize_energy(num_qubits, num_clbits, operations, hamiltonian, initial_parameters, max_iterations, gradient_tolerance, postselection=None, initial_step_size=0.5)` 执行确定性梯度下降：

1. 在当前点计算精确能量、成功概率和梯度。
2. 沿负梯度方向试探；能量上升或试探点后选失败时将步长乘以 0.5 后重试。
3. 接受不增（带 1e-12 相对松弛）能量的点；成功使用完整步长时下一轮允许放大步长，但不超过 `initial_step_size`。
4. 若步长缩小到 1e-12 以下仍无有效试探，保留最后一个有效点并停止。

返回 `TrainingResult`：`parameters`、`energy`、`success_probability`、`gradient`、`history`（每个被接受迭代的参数、能量、概率、梯度范数和实际步长）、`iterations` 和 `stop_reason`。停止原因取值：

- `gradient_tolerance`：梯度范数达到容差，是唯一的收敛（`converged` 为 True）。
- `max_iterations`：迭代预算耗尽，不是收敛。
- `step_size_underflow`：试探全部失败后的停滞，不是收敛；返回最后有效点。

这是局部最小化，不承诺全局最优。初值点后选概率不超过 1e-12 时立即抛出 `PostselectionError`；非法迭代数、容差或步长抛出 `TrainingError`。

## 量子态层析

0.3 版本新增 1 至 3 量子位的完整局部 Pauli 层析，由测量计划、现有分支演化、计数重建和物理态投影多个模块协作完成：

- `pauli_measurement_plan(num_qubits)` 返回 `3**n` 个设置。基串右端对应量子位 0，例如 2 位的 `XZ` 表示量子位 0 用 Z、量子位 1 用 X。
- `simulate_tomography_counts(num_qubits, num_clbits, operations, parameter_values=None, postselection=None, shots=1000, seed=None)` 复用参数绑定和 `simulate_branches` 分支演化。它只读取经典寄存器进行后选，不会运行额外 MEASURE，也不覆写原经典位。
- 后选分支的加权密度矩阵先除以总成功概率得到归一化条件末态，再分别旋转到每个 Pauli 设置的计算基并使用独立的局部 `numpy.random.Generator` 抽样；相同种子可复现，且不污染全局随机状态。
- 后选成功概率不超过 `1e-12` 时抛出 `PostselectionError`。返回的 `TomographyExperiment` 包含 `plan`、`counts`、每个设置的 `shots` 和 `success_probability`。`shots` 可以是所有设置共用的正整数，也可以是 `{设置: 正整数}` 的映射，因此不同设置次数可以不同。
- 结果位串右端同样对应量子位 0；读数 `0` 和 `1` 分别表示该基的正、负本征值。

独立的重建入口 `reconstruct_density_matrix(plan, counts)` 只接收测量计划和计数，不读取模拟器末态：

- 每个设置必须恰好出现一份，计划必须包含完整的 3**n 个 X/Y/Z 设置；重复、缺失或非法基字符抛出 `TomographyError`。
- `counts` 是 `{基串: {结果串: 整数计数}}`。缺失的结果串按零处理；非法结果串字符或长度、负计数、布尔值、非整数计数以及某设置总次数为零都会被拒绝。输入映射不会被修改。
- 对每个 Pauli 串，所有相容设置中未测量量子位被边缘化；重建将相应的奇偶符号计数求和，并按各设置总次数加权。例如 `IZ` 的估计会汇总 `XZ`、`YZ`、`ZZ` 中只看量子位 0 的符号。
- 以 `rho = (1/2**n) sum_P <P> P` 进行线性 Pauli 展开，返回 `TomographyResult`：`expectations`、原始 `density_matrix`、`physical_density_matrix`、原始矩阵 `min_eigenvalue` 和 `correction_distance`（Frobenius 范数）。
- 物理矩阵通过对原始 Hermitian 矩阵做特征值分解、再把特征值投影到概率单纯形得到，是半正定且迹为 1 集合内的 Frobenius 最近投影。

### 统计误差

层析计数服从多项分布。对某一设置中由 m 个被测量量子位定义的奇偶符号，M 次试验的期望估计标准误差不超过 `1/sqrt(M)`；合并多个相容设置时，有效次数为这些设置次数之和。增加每个设置的 shot 数可降低随机误差。有限计数下的原始线性重建虽近似迹为 1 和 Hermitian，但可能出现很小的负特征值；这不是演化错误，而是统计涨落。需要保证合法密度矩阵的后续计算应使用 `physical_density_matrix`，并通过 `min_eigenvalue` 和 `correction_distance` 判断统计修正幅度。

典型用法见 `examples/tomography_after_training.py`：先完成变分训练，再用训练后的参数采集完整 Pauli 计数并重建 2 量子位态。

## 校验

越界量子位或经典位、相同的 CX 控制位和目标位、未知操作、非有限 RZ 角度、非法概率和非法反馈条件都会抛出 `CircuitValidationError`。错误消息包含从 0 开始的操作位置，例如 `operation 3`。校验在分配量子态前完成；SDK 不会返回部分执行结果，也不会修改调用方传入的数据。

## 数值容差

所有矩阵使用 NumPy `complex128`。内部比较用于剪除严格为 0 的概率分支；测试默认使用 `1e-10` 的绝对和相对容差。最多 6 个量子位时密度矩阵为 64×64，确定性演化开销可控。后选判定阈值固定为 1e-12。

## 测试与示例

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python examples/feedback_noise.py
PYTHONPATH=src .venv/bin/python examples/variational_training.py
PYTHONPATH=src .venv/bin/python examples/tomography_after_training.py
```

`feedback_noise.py` 包含 Bell 纠缠、相位翻转噪声、测量以及基于测量结果的条件 X 门。`variational_training.py` 演示一个共享参数（含负缩放）同时驱动两个 RZ 门，在测量反馈和 c0=0 后选下训练条件能量并打印真实停止原因。
