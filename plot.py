#!/usr/bin/env python3
# plot.py

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/benchmark.csv")

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

# ── 1. solvers ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(df["N"], df["ms_jacobi_seq"], "o-",  color="#4C72B0", label="Jacobi sequential")
ax.plot(df["N"], df["ms_jacobi_par"], "o--", color="#4C72B0", alpha=0.6, label="Jacobi parallel")
ax.plot(df["N"], df["ms_gs_par"],     "s--", color="#DD8452", alpha=0.8, label="Gauss-Seidel")

ax.set_xlabel("Matrix size N")
ax.set_ylabel("Time, ms")
ax.set_title("Jacobi vs Gauss-Seidel: solve time")
ax.legend()

plt.tight_layout()
plt.savefig("plots/solvers.png")
plt.close()
print("saved: plots/solvers.png")

# ── 2. matvec ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_yscale("log")
ax.set_xscale("log")

ax.plot(df["N"], df["ms_csr_seq"],     "o-",  color="#4C72B0", label="CSR sequential")
ax.plot(df["N"], df["ms_csr_static"],  "o--", color="#4C72B0", alpha=0.6, label="CSR parallel static")
ax.plot(df["N"], df["ms_csr_dynamic"], "o:",  color="#4C72B0", alpha=0.4, label="CSR parallel dynamic")
ax.plot(df["N"], df["ms_dense_seq"],   "s-",  color="#DD8452", label="Dense sequential")
ax.plot(df["N"], df["ms_dense_par"],   "s--", color="#DD8452", alpha=0.6, label="Dense parallel")

ax.set_xlabel("Matrix size N")
ax.set_ylabel("Time, ms")
ax.set_title("CSR vs Dense matvec")
ax.legend()

plt.tight_layout()
plt.savefig("plots/matvec.png")
plt.close()
print("saved: plots/matvec.png")