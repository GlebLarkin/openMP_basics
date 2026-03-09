import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/benchmark.csv")

plt.figure()

plt.plot(df["N"], df["ms_jacobi_seq"], "o-", label="Jacobi sequential")
plt.plot(df["N"], df["ms_jacobi_par"], "o--", label="Jacobi parallel")
plt.plot(df["N"], df["ms_gs_par"], "s--", label="Gauss-Seidel")

plt.xlabel("Matrix size N")
plt.ylabel("Time, ms")
plt.title("Jacobi vs Gauss-Seidel: solve time")

plt.minorticks_on()
plt.grid(True, which="major", linewidth=0.8)
plt.grid(True, which="minor", linewidth=0.4, alpha=0.5)

plt.legend()
plt.savefig("plots/solvers.png")
plt.close()


plt.figure()

plt.xscale("log")
plt.yscale("log")

plt.plot(df["N"], df["ms_csr_seq"], "o-", label="CSR sequential")
plt.plot(df["N"], df["ms_csr_static"], "o--", label="CSR parallel static")
plt.plot(df["N"], df["ms_csr_dynamic"], "o:", label="CSR parallel dynamic")
plt.plot(df["N"], df["ms_dense_seq"], "s-", label="Dense sequential")
plt.plot(df["N"], df["ms_dense_par"], "s--", label="Dense parallel")

plt.xlabel("Matrix size N")
plt.ylabel("Time, ms")
plt.title("CSR vs Dense matvec")

plt.minorticks_on()
plt.grid(True, which="major", linewidth=0.8)
plt.grid(True, which="minor", linewidth=0.4, alpha=0.5)

plt.legend()
plt.savefig("plots/matvec1.png")
plt.close()