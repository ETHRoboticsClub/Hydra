#!/usr/bin/env bash
set -euo pipefail

RESULTS_FILE="/checkpoints/gpu-test-results.txt"

# ── 1. System info ───────────────────────────────────────────────────────────
echo "=== GPU Smoke Test ==="
echo "Hostname: $(hostname)"
echo "Date:     $(date -u)"
echo "Python:   $(python3 --version 2>&1)"
echo ""

# ── 2. CUDA / GPU check ─────────────────────────────────────────────────────
echo "=== GPU Info ==="
nvidia-smi
echo ""

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
print(f'CUDA version:    {torch.version.cuda}')
print(f'GPU count:       {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print()
"

# ── 3. GPU matrix multiply ──────────────────────────────────────────────────
echo "=== Running GPU matrix multiply ==="
python3 -c "
import torch, time

device = torch.device('cuda')
size = 4096

a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warm-up
torch.mm(a, b)
torch.cuda.synchronize()

# Timed run
start = time.time()
c = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f'Matrix size:  {size}x{size}')
print(f'Compute time: {elapsed:.4f}s')
print(f'Result shape: {c.shape}')
print(f'Result norm:  {c.norm().item():.2f}')
print()
print('GPU smoke test PASSED')
"

# ── 4. Write results to checkpoint volume ────────────────────────────────────
echo "=== Writing results ==="
mkdir -p "$(dirname "${RESULTS_FILE}")"
{
  echo "gpu-smoke-test results"
  echo "date: $(date -u)"
  echo "hostname: $(hostname)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  python3 -c "
import torch
print(f'pytorch: {torch.__version__}')
print(f'cuda: {torch.version.cuda}')
print(f'gpu_count: {torch.cuda.device_count()}')
print('status: PASSED')
"
} > "${RESULTS_FILE}"

echo "Results written to ${RESULTS_FILE}"
echo "=== Done ==="
