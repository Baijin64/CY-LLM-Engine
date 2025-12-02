<!-- CONTRIBUTING.md -->

# 贡献指南 (Contributing)

欢迎为本项目做贡献！为保持仓库整洁并便于协作，请遵循下列指南。

## 代码提交与版本号规范
- 项目采用 **四段式版本号**：`[major.minor.patch.build-SUFFIX]`，例如 `[2.1.1.2-Alpha]`。
- 提交时：先提交英文描述（标题 + 正文），然后空一行再添加中文翻译。
- 标题格式：`[version] <type(scope): short description>`，例如：
  - `[2.1.1.2-Alpha] feat(worker): add async infer support`（英文）
  - 空一行
  - `【中文】worker：添加异步推理支持`
- 若变更包含破坏性 API 修改，请提升 major 段并在正文中说明。

## 提交流程
1. Fork 仓库并在 feature 分支上提交（`feature/xxx`，`fix/xxx`）
2. 编写单元测试，确保公共接口稳定
3. 按照项目测试指南运行单元/集成测试
4. 提交并在 PR 描述中说明兼容性、需要 REVIEW 的点与测试策略

## 测试要求
 Python: 使用 `pytest`，测试目录：`CY_LLM_Backend/worker/tests`。
## 风险与回退策略

## 代码风格与质量
- Python: 遵循 PEP8，建议使用 `black`、`ruff`、`mypy`。
- Kotlin: 遵循 Kotlin 官方编码规范，项目启用了 `ktlint` 与 `detekt`。

## 联系与支持
有任何问题可在 Issue 里提出，我们会尽快回复。
