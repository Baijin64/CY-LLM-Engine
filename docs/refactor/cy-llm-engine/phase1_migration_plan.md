# Phase 1: 目录合并分析与迁移计划

## 1. 重复分析总结

### 目录对比统计

| 目录 | src/cy_llm/ | CY_LLM_Backend/ | 状态 |
|------|-------------|-----------------|------|
| worker/tests/ | 17文件 | 17文件 | 完全重复 |
| worker/training/ | 9文件 | 9文件 | 完全重复 |
| worker/utils/ | 8文件 | 8文件 | 完全重复 |
| worker/engines/ | 10文件 | 10文件 | 完全重复 |
| worker/core/ | 4文件 | 4文件 | 完全重复 |
| worker/config/ | 5文件 | 5文件 | 完全重复 |
| worker/cache/ | 2文件 | 2文件 | 完全重复 |
| worker/health/ | 2文件 | 2文件 | 完全重复 |
| worker/proto_gen/ | 3文件 | 3文件 | 完全重复 |
| worker/*.py | 7文件 | 7文件 | 完全重复 |
| coordinator_lite/ | 3文件 | 3文件 | 完全重复 |
| gateway_lite/ | 0文件 | 3文件 | 仅CY_LLM_Backend有 |
| training/ | 0文件 | 4文件 | 仅CY_LLM_Backend有 |

### 结论
**99%的文件是完全重复的**，且CY_LLM_Backend包含更多文件（gateway_lite, training等）。

### CLI脚本验证
cy-llm脚本使用路径：`BACKEND_DIR="$SCRIPT_DIR/CY_LLM_Backend"`

**结论**: CY_LLM_Backend是活跃目录，src/cy_llm是老版本。

## 2. 迁移策略

### 策略: 保留CY_LLM_Backend，删除src/cy_llm/

**原因**:
1. CLI脚本指向CY_LLM_Backend
2. CY_LLM_Backend文件更完整
3. README中示例使用CY_LLM_Backend路径
4. src/cy_llm可能是遗留代码

### 迁移计划

#### Step 1: 备份src/cy_llm（以防万一）
```bash
cp -r src/cy_llm src/cy_llm.backup.$(date +%Y%m%d)
```

#### Step 2: 检查src/cy_llm中是否有CY_LLM_Backend没有的独特文件
通过文件列表对比，确认所有文件都已存在。

#### Step 3: 更新pyproject.toml
修改package指向：
```toml
[tool.setuptools.packages.find]
where = ["CY_LLM_Backend"]
```

#### Step 4: 创建兼容性软链接（可选，用于平滑迁移）
```bash
ln -s CY_LLM_Backend/src/cy_llm src/cy_llm_compat
```

#### Step 5: 删除src/cy_llm目录
```bash
rm -rf src/cy_llm
```

#### Step 6: 更新所有import路径
将 `from src.cy_llm...` 改为 `from CY_LLM_Backend...` 或直接 `from cy_llm...`

#### Step 7: 测试验证
```bash
python -c "from CY_LLM_Backend.worker import main"
```

## 3. 详细文件映射

### 完全相同的文件（可直接删除src版本）
- worker/main.py
- worker/__init__.py
- worker/constants.py
- worker/exceptions.py
- worker/grpc_servicer.py
- worker/grpc_servicer_async.py
- worker/training_engine.py
- worker/training_servicer_grpc.py
- worker/REFACTORING.py
- worker/core/*.py (4 files)
- worker/engines/*.py (10 files)
- worker/config/*.py (5 files)
- worker/utils/*.py (8 files)
- worker/cache/*.py (2 files)
- worker/health/*.py (2 files)
- worker/tests/*.py (17 files)
- worker/training/**/*.py (9 files)
- worker/proto_gen/*.py (3 files)
- coordinator_lite/**/*.py (3 files)

### CY_LLM_Backend独有的文件（保留）
- gateway_lite/app/main.py
- gateway_lite/app/__init__.py
- gateway_lite/__init__.py
- training/engine.py
- training/__init__.py
- training/model/*.py
- tests/test_integration.py

### src/cy_llm独有的文件
无（所有文件在CY_LLM_Backend中都存在）

## 4. 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 删除错误 | 🔴 High | 先备份，git保留历史 |
| Import路径失效 | 🟡 Medium | 全局替换import语句 |
| 测试失败 | 🟡 Medium | 先运行基线测试 |
| 文档过期 | 🟢 Low | 同步更新README |

## 5. 回滚计划

如果迁移后出现问题：
```bash
# 从备份恢复
cp -r src/cy_llm.backup.20260210 src/cy_llm

# 或从git恢复
git checkout HEAD -- src/cy_llm
```

## 6. 执行检查清单

- [ ] 备份src/cy_llm
- [ ] 验证无独有文件
- [ ] 更新pyproject.toml
- [ ] 删除src/cy_llm
- [ ] 更新import路径
- [ ] 运行单元测试
- [ ] 更新文档

## 7. 实施顺序

1. **T-020**: 完成目录重复分析 ✅
2. **T-021**: 迁移计划获批 ✅
3. **T-022**: 合并核心模块（实际上无需合并，直接删除重复）
4. **T-023**: 合并引擎模块（同上）
5. **T-024**: 合并配置和工具（同上）
6. **T-025**: 删除src/cy_llm目录

**简化**: 由于完全重复，无需复杂合并，直接删除src/cy_llm即可。
