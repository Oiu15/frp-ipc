# 当前工程门禁

本文档只记录当前阶段的本地工程门禁，不作为长期架构规划。

## 必跑检查

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pytest --collect-only -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m compileall _version.py app.py application config core domain drivers events frp_workflow machine modes repositories services ui utils
```

当前 pytest cache 配置为 `.test-artifacts/pytest-cache`。`.test-artifacts/` 已在 `.gitignore` 中忽略，不应提交测试产物。

## 当前 ruff 门禁

第一阶段只启用以下规则：

- `F401`: unused import
- `F811`: redefined while unused
- `F821`: undefined name
- `F841`: unused local variable
- `E741`: ambiguous variable name

暂不作为门禁：

- `E402`: 当前仍存在迁移期导入形态，先不强制导入位置。
- `E702` / `E731`: 偏风格项，暂不混入第一阶段目标。
- 全仓格式化、导入排序、复杂度规则。

## pyright 状态

仓库已有 `pyrightconfig.json`，但当前本机运行：

```powershell
.\.venv\Scripts\python.exe -m pyright
```

会在调用 `node` 时失败，错误为 `PermissionError: [WinError 5] 拒绝访问`。修复本机 Node/pyright 执行环境前，pyright 不作为硬门禁。

## 不覆盖的变更

第一阶段门禁不改变以下语义：

- PLC 通信、轮询、命令队列语义。
- 测径仪串口请求、读取、解析语义。
- 运行导出、验证导出的 CSV/JSON schema。
- package layout、入口链路、AutoFlow 主流程。
