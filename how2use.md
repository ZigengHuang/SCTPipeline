# 一键批量 → 自动 domain → 断点续跑 → 日志 → 汇总报告
# 用法示例

## 批量运行（默认 `domain=1`）
```bash
  python assistant_agent.py \
  --input ./input_ructs \
  --output ./output_scts \
  --task "Please denoise, correct and identify speaker this clinical note"
```

## 指定领域（例如 `domain=3` → 治疗记录）
```bash
  python assistant_agent.py \
  --input ./input_treatment \
  --output ./output_scts \
  --task "Please denoise, correct and identify speaker this clinical note" \
  --domain 3
```

## 断点续跑模式
```bash
  python assistant_agent.py \
  --input ./input_ructs \
  --output ./output_scts \
  --task "Please denoise, correct and identify speaker this clinical note" \
  --domain 1 \
  --resume
```
