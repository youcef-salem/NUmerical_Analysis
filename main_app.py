import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import importlib.util, sys, os

# ── load algo modules ──────────────────────────────────────────────────────────
def load_module(name, path):
    mod_dir = os.path.dirname(os.path.abspath(path))
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BASE = os.path.dirname(os.path.abspath(__file__))

dichotomie_mod        = load_module("dichotomie",               os.path.join(BASE, "axe_1/dichotomie.py"))
newton_mod            = load_module("newton",                   os.path.join(BASE, "axe_1/newton.py"))
point_fixe_mod        = load_module("point_fixe",               os.path.join(BASE, "axe_1/point_fixe.py"))
gauss_mod             = load_module("methode_gausse",           os.path.join(BASE, "axe_2/methode_gausse.py"))
lu_mod                = load_module("methode_decomposition_LU", os.path.join(BASE, "axe_2/methode_decomposition_LU.py"))
cholesky_mod          = load_module("cholevsky",                os.path.join(BASE, "axe_2/cholevsky.py"))
lagrange_mod          = load_module("interpolation_lagrange",   os.path.join(BASE, "axe_3/interpolation_lagrange.py"))
newton_interp_mod     = load_module("interpolation_newton",     os.path.join(BASE, "axe_3/interpolation_newton.py"))
descente_gradient_mod = load_module("descente_gradient",        os.path.join(BASE, "axe_3/descente_gradient.py"))
normes_mod            = load_module("normes",                   os.path.join(BASE, "axe_2/normes.py"))


# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numerical Methods")
        self.geometry("800x560")
        self.resizable(True, True)
        self._show_welcome()

    def clear(self):
        for w in self.winfo_children():
            w.destroy()

    # ── Page d'acceuil ───────────────────────────────────────────────────────────────
    def _show_welcome(self):
        self.clear()
        f = tk.Frame(self)
        f.place(relx=.5, rely=.5, anchor="center")

        tk.Label(f, text="Numerical Methods", font=("TkDefaultFont", 22, "bold")).pack(pady=(0, 4))
        tk.Label(f, text="Phase 1  —  Axes 1 to 3", font=("TkDefaultFont", 11)).pack()

        tk.Frame(f, height=60).pack()

        tk.Button(f, text="Begin", width=16, command=self._show_menu).pack(pady=8)

    # ── menu axes ──────────────────────────────────────────────────────────────────
    def _show_menu(self):
        self.clear()
        f = tk.Frame(self)
        f.place(relx=.5, rely=.5, anchor="center")

        tk.Label(f, text="Select an option", font=("TkDefaultFont", 14, "bold")).pack(pady=(0, 20))

        options = [
            ("Axe 1 - Resolution de equations non lineaires", self._show_axis1),
            ("Axe 2 - Resolution d'equations linéaires",      self._show_axis2),
            ("Axe 3 - Interpolation & Approximation",         self._show_axis3),
        ]
        for text, cmd in options:
            tk.Button(f, text=text, width=56, anchor="w", padx=8, command=cmd).pack(pady=4)

    # ══════════════════════════════════════════════════════════════════════════
    # AXE 1
    # ══════════════════════════════════════════════════════════════════════════
    def _show_axis1(self):
        self.clear()

        canvas = tk.Canvas(self, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        outer = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=outer, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        outer.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        p = dict(padx=20, pady=6)

        tk.Label(outer, text="Axe 1 - Resolution de equations non lineaires",
                 font=("TkDefaultFont", 13, "bold")).pack(anchor="w", **p)
        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20)

        fx_frame = tk.LabelFrame(outer, text="Function", padx=10, pady=8)
        fx_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(fx_frame, text="f(x) =").grid(row=0, column=0, sticky="w")
        self.fx_entry = tk.Entry(fx_frame, width=34)
        self.fx_entry.grid(row=0, column=1, padx=6)

        btn_row = tk.Frame(fx_frame)
        btn_row.grid(row=0, column=2, padx=10)
        tk.Button(btn_row, text="Dessiner graphe",        command=self._ax1_draw_graph).pack(side="left", padx=3)
        tk.Button(btn_row, text="Vérifier la continuité", command=self._ax1_continuity).pack(side="left", padx=3)
        tk.Button(btn_row, text="Dérivée",                command=self._ax1_derivative).pack(side="left", padx=3)

        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20)

        tk.Label(outer, text="Methods", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=20, pady=(8,2))

        mrow = tk.Frame(outer)
        mrow.pack(anchor="w", padx=20, pady=4)

        # Dichotomie
        dc = tk.LabelFrame(mrow, text="Dichotomie", padx=10, pady=8)
        dc.pack(side="left", padx=(0,12), anchor="n")
        r = tk.Frame(dc); r.pack(anchor="w")
        tk.Label(r, text="a:").pack(side="left")
        self.dich_a = tk.Entry(r, width=7); self.dich_a.pack(side="left", padx=3)
        tk.Label(r, text="b:").pack(side="left")
        self.dich_b = tk.Entry(r, width=7); self.dich_b.pack(side="left", padx=3)
        tk.Button(dc, text="Run", command=self._ax1_dichotomie).pack(pady=(6,0))

        # Point Fixe
        pfc = tk.LabelFrame(mrow, text="Point Fixe", padx=10, pady=8)
        pfc.pack(side="left", padx=(0,12), anchor="n")
        tk.Label(pfc, text="g(x) =").pack(anchor="w")
        self.gx_entry = tk.Entry(pfc, width=20); self.gx_entry.pack(anchor="w", pady=2)
        r2 = tk.Frame(pfc); r2.pack(anchor="w")
        tk.Label(r2, text="x0:").pack(side="left")
        self.pf_x0 = tk.Entry(r2, width=7); self.pf_x0.pack(side="left", padx=3)
        tk.Button(pfc, text="Run", command=self._ax1_point_fixe).pack(pady=(6,0))

        # Newton
        nwc = tk.LabelFrame(mrow, text="Newton", padx=10, pady=8)
        nwc.pack(side="left", anchor="n")
        r3 = tk.Frame(nwc); r3.pack(anchor="w")
        tk.Label(r3, text="x0:").pack(side="left")
        self.nw_x0 = tk.Entry(r3, width=7); self.nw_x0.pack(side="left", padx=3)
        tk.Button(nwc, text="Run", command=self._ax1_newton).pack(pady=(6,0))

        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20, pady=10)
        tk.Button(outer, text="← Retour", command=self._show_menu).pack(anchor="e", padx=20, pady=6)

    # ── fonctions utiles ────────────────────────────────────────────────────────
    def _parse_f(self):
        expr = self.fx_entry.get().strip()
        if not expr:
            messagebox.showerror("Error", "Entrer une valeur pour f(x)")
            return None
        try:
            f = lambda x: eval(expr, {"x": x, "np": np,
                                      "sin": np.sin, "cos": np.cos,
                                      "exp": np.exp, "log": np.log,
                                      "sqrt": np.sqrt, "tan": np.tan,
                                      "pi": np.pi, "e": np.e})
            f(1.0)
            return f
        except Exception as ex:
            messagebox.showerror("Error", f"Bad expression: {ex}")
            return None

    def _draw_f_on(self, ax, f, x_range=(-10, 10), root=None):
        xs = np.linspace(x_range[0], x_range[1], 600)
        ys = []
        for x in xs:
            try:
                y = f(x)
                ys.append(float(y) if np.isfinite(y) else np.nan)
            except:
                ys.append(np.nan)
        ax.plot(xs, ys, label=f"f(x) = {self.fx_entry.get()}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        if root is not None:
            ax.axvline(root, color="tab:red", linestyle="--", label=f"root ≈ {root:.6f}")
        ax.set_xlabel("x"); ax.set_ylabel("f(x)")
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

    def _make_method_figure(self):
        fig, (ax_graph, ax_table) = plt.subplots(
            2, 1, figsize=(8, 8),
            gridspec_kw={"height_ratios": [2, 1.4]}
        )
        return fig, ax_graph, ax_table

    # ── options additionnelles ───────────────────────────────
    def _ax1_draw_graph(self):
        f = self._parse_f()
        if not f: return
        fig, ax = plt.subplots()
        self._draw_f_on(ax, f)
        ax.set_title("Graph of f(x)")
        plt.tight_layout(); plt.show()

    def _ax1_continuity(self):
        f = self._parse_f()
        if not f: return
        xs = np.linspace(-10, 10, 1000)
        try:
            ys = np.array([f(x) for x in xs], dtype=float)
            jumps = np.where(np.abs(np.diff(ys)) > 50)[0]
            if len(jumps) == 0:
                messagebox.showinfo("Continuity", "f appears continuous on [-10, 10].")
            else:
                pts = ", ".join(f"{xs[j]:.3f}" for j in jumps[:5])
                messagebox.showwarning("Continuity", f"Possible discontinuity near: {pts}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    def _ax1_derivative(self):
        f = self._parse_f()
        if not f: return
        h = 1e-5
        df = lambda x: (f(x+h) - f(x-h)) / (2*h)
        xs = np.linspace(-10, 10, 400)
        try:
            ys = [df(x) for x in xs]
        except:
            messagebox.showerror("Error", "Could not evaluate derivative"); return
        fig, ax = plt.subplots()
        ax.plot(xs, ys, color="tab:red", label="f'(x)  [numerical]")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Derivative of f(x)"); ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout(); plt.show()

    # ── Dichotomie ────────────────────────────────────────────────────────────
    def _ax1_dichotomie(self):
        f = self._parse_f()
        if not f: return
        try:
            a = float(self.dich_a.get()); b = float(self.dich_b.get())
        except:
            messagebox.showerror("Error", "Invalid a or b"); return

        fig, ax_graph, ax_table = self._make_method_figure()
        fig.suptitle("Dichotomie")
        try:
            root = dichotomie_mod.dichotomie(f, a, b, display=True, display_plot=ax_table)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); plt.close(fig); return

        self._draw_f_on(ax_graph, f, x_range=(a - 1, b + 1), root=root)
        ax_graph.set_title(f"f(x)  —  root ≈ {root:.8f}")
        ax_table.set_title("Iteration table", pad=4)
        plt.tight_layout(); plt.show()

    # ── Point Fixe ────────────────────────────────────────────────────────────
    def _ax1_point_fixe(self):
        f = self._parse_f()
        if not f: return
        gx_expr = self.gx_entry.get().strip()
        if not gx_expr:
            messagebox.showerror("Error", "Entrer une valeur pour g(x)"); return
        try:
            g = lambda x: eval(gx_expr, {"x": x, "np": np,
                                          "sin": np.sin, "cos": np.cos,
                                          "exp": np.exp, "log": np.log,
                                          "sqrt": np.sqrt, "pi": np.pi})
            x0 = float(self.pf_x0.get())
        except:
            messagebox.showerror("Error", "Invalid g(x) or x0"); return

        fig, ax_graph, ax_table = self._make_method_figure()
        fig.suptitle("Point Fixe")
        try:
            I, X, X_next, errs = point_fixe_mod.fixed_point_iteration(
                g, x0, precision=1e-10, display=True, display_plot=ax_table)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); plt.close(fig); return

        root = X_next[-1] if X_next else x0
        self._draw_f_on(ax_graph, f, root=root)
        for xn in X_next[:30]:
            try: ax_graph.plot(xn, f(xn), "o", color="tab:orange", markersize=3, alpha=0.6)
            except: pass
        ax_graph.set_title(f"f(x)  —  root ≈ {root:.8f}  (orange = iterates)")
        ax_table.set_title("Iteration table", pad=4)
        plt.tight_layout(); plt.show()

    # ── Newton ────────────────────────────────────────────────────────────────
    def _ax1_newton(self):
        f = self._parse_f()
        if not f: return
        try:
            x0 = float(self.nw_x0.get())
        except:
            messagebox.showerror("Error", "Entrer une valeur pour x0"); return

        fig, ax_graph, ax_table = self._make_method_figure()
        fig.suptitle("Newton")
        try:
            xs = newton_mod.newton(f, x0, precision=1e-10, display=True, display_plot=ax_table)
        except Exception as ex:
            messagebox.showerror("Error", str(ex)); plt.close(fig); return

        root = xs[-1]
        self._draw_f_on(ax_graph, f, root=root)
        for xn in xs[:-1]:
            try: ax_graph.plot(xn, f(xn), "o", color="tab:orange", markersize=4, alpha=0.7)
            except: pass
        ax_graph.set_title(f"f(x)  —  root ≈ {root:.8f}  (orange = iterates)")
        ax_table.set_title("Iteration table", pad=4)
        plt.tight_layout(); plt.show()

    # ══════════════════════════════════════════════════════════════════════════
    # AXE 2
    # ══════════════════════════════════════════════════════════════════════════
    def _show_axis2(self):
        self.clear()

        canvas = tk.Canvas(self, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        outer = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=outer, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        outer.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        p = dict(padx=20, pady=6)

        tk.Label(outer, text="Axe 2 — Systèmes Linéaires",
                 font=("TkDefaultFont", 13, "bold")).pack(anchor="w", **p)
        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20)

        # ── dimension + grilles + normes (même ligne) ──────────────────────
        top_row = tk.Frame(outer)
        top_row.pack(anchor="w", padx=20, pady=10, fill="x")

        # colonne gauche : n + grilles
        left_col = tk.Frame(top_row)
        left_col.pack(side="left", anchor="n")

        nf = tk.Frame(left_col)
        nf.pack(anchor="w")
        tk.Label(nf, text="Dimension  n =").pack(side="left")
        self.n_var = tk.StringVar(value="3")
        tk.Spinbox(nf, from_=1, to=10, textvariable=self.n_var, width=4).pack(side="left", padx=6)
        tk.Button(nf, text="Update grids", command=self._ax2_rebuild_grids).pack(side="left", padx=6)

        self.ax2_grid_frame = tk.Frame(left_col)
        self.ax2_grid_frame.pack(anchor="w", pady=4)
        self.ax2_A_cells = []
        self.ax2_b_cells = []
        self._ax2_build_grids(3)

        # colonne droite : normes induites
        norms_frame = tk.LabelFrame(top_row, text="Normes induites", padx=10, pady=8)
        norms_frame.pack(side="left", anchor="n", padx=(30, 0))

        tk.Label(norms_frame, text="Calculer ||A|| pour la\nmatrice A saisie ci-contre",
                 justify="left").pack(anchor="w", pady=(0, 6))
        tk.Button(norms_frame, text="Norme 1  (max col)",   width=22,
                  command=lambda: self._ax2_norme(1)).pack(pady=2)
        tk.Button(norms_frame, text="Norme ∞  (max ligne)", width=22,
                  command=lambda: self._ax2_norme("inf")).pack(pady=2)
        tk.Button(norms_frame, text="Norme 2  (spectrale)", width=22,
                  command=lambda: self._ax2_norme(2)).pack(pady=2)

        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20, pady=4)

        # ── méthodes directes ──────────────────────────────────────────────
        tk.Label(outer, text="Méthode directe", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=20)
        df = tk.Frame(outer); df.pack(anchor="w", padx=20, pady=4)
        tk.Button(df, text="Elimination de Gauss", command=self._ax2_gauss).pack(side="left", padx=4)
        tk.Button(df, text="Décomposition LU",     command=self._ax2_lu).pack(side="left", padx=4)
        tk.Button(df, text="Cholesky",             command=self._ax2_cholesky).pack(side="left", padx=4)

        # ── méthodes indirectes ────────────────────────────────────────────
        tk.Label(outer, text="Méthode indirecte", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=20, pady=(10,0))
        inf2 = tk.Frame(outer); inf2.pack(anchor="w", padx=20, pady=4)
        tk.Button(inf2, text="Jacobi",       command=self._ax2_jacobi).pack(side="left", padx=4)
        tk.Button(inf2, text="Gauss-Seidel", command=self._ax2_gauss_seidel).pack(side="left", padx=4)

        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20, pady=10)
        tk.Button(outer, text="← Retour", command=self._show_menu).pack(anchor="e", padx=20, pady=6)

    # ── entrée utilisateur ─────────────────────────────────────────────────
    def _ax2_build_grids(self, n):
        for w in self.ax2_grid_frame.winfo_children():
            w.destroy()
        self.ax2_A_cells = [[None]*n for _ in range(n)]
        self.ax2_b_cells = [None]*n

        tk.Label(self.ax2_grid_frame, text="A =").grid(row=0, column=0, rowspan=n, sticky="e", padx=(0,6))
        for i in range(n):
            for j in range(n):
                e = tk.Entry(self.ax2_grid_frame, width=6, justify="center")
                e.insert(0, "0"); e.grid(row=i, column=j+1, padx=2, pady=2)
                self.ax2_A_cells[i][j] = e

        tk.Label(self.ax2_grid_frame, text="b =").grid(row=0, column=n+1, rowspan=n, sticky="e", padx=(14,6))
        for i in range(n):
            e = tk.Entry(self.ax2_grid_frame, width=6, justify="center")
            e.insert(0, "0"); e.grid(row=i, column=n+2, padx=2, pady=2)
            self.ax2_b_cells[i] = e

    def _ax2_rebuild_grids(self):
        try: n = max(1, min(10, int(self.n_var.get())))
        except: return
        self._ax2_build_grids(n)

    def _ax2_read_system(self):
        n = len(self.ax2_A_cells)
        A, b = [], []
        for i in range(n):
            row = []
            for j in range(n):
                try: row.append(float(self.ax2_A_cells[i][j].get()))
                except: messagebox.showerror("Error", f"Bad value A[{i}][{j}]"); return None, None
            A.append(row)
            try: b.append(float(self.ax2_b_cells[i].get()))
            except: messagebox.showerror("Error", f"Bad value b[{i}]"); return None, None
        return A, b

    def _result_window(self, title, text):
        w = tk.Toplevel(self)
        w.title(title); w.geometry("400x300")
        tk.Label(w, text=title, font=("TkDefaultFont", 12, "bold")).pack(pady=8)
        t = tk.Text(w, font=("Courier New", 10))
        t.pack(fill="both", expand=True, padx=12, pady=(0,8))
        t.insert("1.0", text); t.config(state="disabled")
        tk.Button(w, text="Close", command=w.destroy).pack(pady=6)

    # ── Normes induites ────────────────────────────────────────────────────
    def _ax2_norme(self, kind):
        A, _ = self._ax2_read_system()
        if A is None: return
        try:
            if kind == 1:
                val = normes_mod.norme_1(A)
                label = "Norme induite 1  ||A||₁  (max somme colonnes)"
            elif kind == "inf":
                val = normes_mod.norme_inf(A)
                label = "Norme induite ∞  ||A||∞  (max somme lignes)"
            else:
                val = normes_mod.norme_2(A)
                label = "Norme induite 2  ||A||₂  (norme spectrale)"
            self._result_window(label, f"{label}\n\n  ||A|| = {val:.10f}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

    # ── Gauss ─────────────────────────────────────────────────────────────
    def _ax2_gauss(self):
        A, b = self._ax2_read_system()
        if A is None: return
        try:
            x = gauss_mod.gaussian_elimination_pp(A, b)
            self._result_window("Elimination de Gauss", "Solution x:\n" +
                                "\n".join(f"  x[{i}] = {v:.8f}" for i,v in enumerate(x)))
        except Exception as ex: messagebox.showerror("Error", str(ex))

    # ── LU ────────────────────────────────────────────────────────────────
    def _ax2_lu(self):
        A, b = self._ax2_read_system()
        if A is None: return
        try:
            x = lu_mod.solve_lu_crout_pp(A, b)
            self._result_window("Decomposition LU", "Solution x:\n" +
                                "\n".join(f"  x[{i}] = {v:.8f}" for i,v in enumerate(x)))
        except Exception as ex: messagebox.showerror("Error", str(ex))

    # ── Cholesky ──────────────────────────────────────────────────────────
    def _ax2_cholesky(self):
        A, b = self._ax2_read_system()
        if A is None: return
        try:
            A_np = np.array(A, dtype=float)
            L = cholesky_mod.cholesky(A_np)
            if L is None:
                messagebox.showerror("Cholesky", "La matrice n'est pas définie positive"); return
            y = np.linalg.solve(L, np.array(b, dtype=float))
            x = np.linalg.solve(L.T, y)
            L_str = "\n".join("  " + "  ".join(f"{L[i,j]:8.4f}" for j in range(len(L)))
                              for i in range(len(L)))
            self._result_window("Cholesky", f"L =\n{L_str}\n\nSolution x:\n" +
                                "\n".join(f"  x[{i}] = {v:.8f}" for i,v in enumerate(x)))
        except Exception as ex: messagebox.showerror("Error", str(ex))

    # ── Jacobi ────────────────────────────────────────────────────────────
    def _ax2_jacobi(self):
        import jaobi as jacobi_mod
        A, b = self._ax2_read_system()
        if A is None: return
        try:
            x, iters = jacobi_mod.jacobi(A, b)
            self._result_window("Jacobi",
                                f"Convergé en {iters} itérations\n\nSolution x:\n" +
                                "\n".join(f"  x[{i}] = {v:.8f}" for i,v in enumerate(x)))
        except Exception as ex: messagebox.showerror("Error", str(ex))

    # ── Gauss-Seidel ──────────────────────────────────────────────────────
    def _ax2_gauss_seidel(self):
        import gauss_seidel as gs_mod
        A, b = self._ax2_read_system()
        if A is None: return
        try:
            x, iters = gs_mod.gauss_seidel(A, b)
            self._result_window("Gauss-Seidel",
                                f"Convergé en {iters} itérations\n\nSolution x:\n" +
                                "\n".join(f"  x[{i}] = {v:.8f}" for i,v in enumerate(x)))
        except Exception as ex: messagebox.showerror("Error", str(ex))

    # ══════════════════════════════════════════════════════════════════════════
    # AXE 3 — Interpolation & Approximation
    # ══════════════════════════════════════════════════════════════════════════
    def _show_axis3(self):
        self.clear()

        canvas = tk.Canvas(self, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        outer = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=outer, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        outer.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        p = dict(padx=20, pady=6)

        tk.Label(outer, text="Axe 3 — Interpolation & Approximation",
                 font=("TkDefaultFont", 13, "bold")).pack(anchor="w", **p)
        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20)

        # ── saisie des points ──────────────────────────────────────────────
        pts_frame = tk.LabelFrame(outer, text="Points de données", padx=10, pady=10)
        pts_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(pts_frame, text="Valeurs de x  (séparées par des virgules) :").pack(anchor="w")
        self.ax3_x_entry = tk.Entry(pts_frame, width=60)
        self.ax3_x_entry.pack(anchor="w", pady=(2, 8))

        tk.Label(pts_frame, text="Valeurs de y  (séparées par des virgules) :").pack(anchor="w")
        self.ax3_y_entry = tk.Entry(pts_frame, width=60)
        self.ax3_y_entry.pack(anchor="w", pady=2)

        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20)

        # ── boutons des méthodes ───────────────────────────────────────────
        tk.Label(outer, text="Méthodes", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=20, pady=(10, 2))

        methods_row = tk.Frame(outer)
        methods_row.pack(anchor="w", padx=20, pady=6)

        interp_frame = tk.LabelFrame(methods_row, text="Interpolation", padx=10, pady=8)
        interp_frame.pack(side="left", padx=(0, 16), anchor="n")
        tk.Button(interp_frame, text="Interpolation de Lagrange", width=26,
                  command=self._ax3_lagrange).pack(pady=3)
        tk.Button(interp_frame, text="Interpolation de Newton",   width=26,
                  command=self._ax3_newton_interp).pack(pady=3)

        approx_frame = tk.LabelFrame(methods_row, text="Approximation", padx=10, pady=8)
        approx_frame.pack(side="left", padx=(0, 16), anchor="n")

        deg_row = tk.Frame(approx_frame); deg_row.pack(anchor="w", pady=(0,4))
        tk.Label(deg_row, text="Degré du polynôme :").pack(side="left")
        self.ax3_deg_var = tk.StringVar(value="2")
        tk.Spinbox(deg_row, from_=1, to=10, textvariable=self.ax3_deg_var, width=4).pack(side="left", padx=4)

        tk.Button(approx_frame, text="Moindres carrés", width=26,
                  command=self._ax3_moindres_carres).pack(pady=3)

        gd_frame = tk.LabelFrame(methods_row, text="Descente de gradient", padx=10, pady=8)
        gd_frame.pack(side="left", anchor="n")

        alpha_row = tk.Frame(gd_frame); alpha_row.pack(anchor="w", pady=(0, 4))
        tk.Label(alpha_row, text="Pas (α) :").pack(side="left")
        self.ax3_alpha_entry = tk.Entry(alpha_row, width=8)
        self.ax3_alpha_entry.insert(0, "0.05")
        self.ax3_alpha_entry.pack(side="left", padx=4)

        deg_row2 = tk.Frame(gd_frame); deg_row2.pack(anchor="w", pady=(0, 4))
        tk.Label(deg_row2, text="Degré :").pack(side="left")
        self.ax3_gd_deg_var = tk.StringVar(value="1")
        tk.Spinbox(deg_row2, from_=1, to=6, textvariable=self.ax3_gd_deg_var, width=4).pack(side="left", padx=4)

        tk.Button(gd_frame, text="Descente de gradient", width=26,
                  command=self._ax3_gradient).pack(pady=3)

        ttk.Separator(outer, orient="horizontal").pack(fill="x", padx=20, pady=10)
        tk.Button(outer, text="← Retour", command=self._show_menu).pack(anchor="e", padx=20, pady=6)

    # ── helpers Axe 3 ─────────────────────────────────────────────────────
    def _ax3_parse_points(self):
        """Lit et valide les champs X et Y. Retourne (x_arr, y_arr) ou (None, None)."""
        try:
            xs = [float(v.strip()) for v in self.ax3_x_entry.get().split(",") if v.strip()]
            ys = [float(v.strip()) for v in self.ax3_y_entry.get().split(",") if v.strip()]
        except ValueError:
            messagebox.showerror("Erreur", "Les valeurs x et y doivent être des nombres séparés par des virgules.")
            return None, None
        if len(xs) < 2:
            messagebox.showerror("Erreur", "Entrer au moins 2 points."); return None, None
        if len(xs) != len(ys):
            messagebox.showerror("Erreur", f"Nombre de x ({len(xs)}) ≠ nombre de y ({len(ys)})."); return None, None
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def _ax3_base_plot(self, ax, x, y, title=""):
        """Dessine les points originaux sur un axe."""
        ax.plot(x, y, "ko", markersize=6, label="Point de la fonction orginiale", zorder=5)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, linestyle="--", alpha=0.5)

    # ── Interpolation Lagrange ────────────────────────────────────────────
    def _ax3_lagrange(self):
        x, y = self._ax3_parse_points()
        if x is None: return

        xx = np.linspace(x[0], x[-1], 400)
        try:
            yy = np.array([lagrange_mod.polynome_de_lagrange(x, y, t) for t in xx])
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex)); return

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8),
                                              gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle("Interpolation de Lagrange")

        # graphe
        self._ax3_base_plot(ax_top, x, y, "Polynôme de Lagrange")
        ax_top.plot(xx, yy, "b-", label="Polynôme de Lagrange")
        ax_top.legend()

        # table des différences (valeurs interpolées aux nœuds)
        ax_bot.axis("tight"); ax_bot.axis("off")
        y_interp = [lagrange_mod.polynome_de_lagrange(x, y, xi) for xi in x]
        errs = [abs(y[i] - y_interp[i]) for i in range(len(x))]
        cell_data = list(zip( [f"{xi:.4f}" for xi in x],
                             [f"{yi:.4f}" for yi in y],
                             [f"{yp:.4f}" for yp in y_interp],
                             [f"{e:.2e}" for e in errs]))
        tbl = ax_bot.table(cellText=cell_data,
                           colLabels=[ "x", "y", "P(x)", "Erreur"],
                           loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        ax_bot.set_title("Vérification aux nœuds", pad=4)

        plt.tight_layout(); plt.show()

    # ── Interpolation Newton ──────────────────────────────────────────────
    def _ax3_newton_interp(self):
        x, y = self._ax3_parse_points()
        if x is None: return

        xx = np.linspace(x[0], x[-1], 400)
        try:
            yy = newton_interp_mod.eval_newton(x, y, xx)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex)); return

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8),
                                              gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle("Interpolation de Newton")

        # graphe
        self._ax3_base_plot(ax_top, x, y, "Polynôme de Newton (différences divisées)")
        ax_top.plot(xx, yy, "b-", label="Polynôme de Newton")
        ax_top.legend()

        # table des différences divisées (première ligne = coefficients)
        tab = newton_interp_mod.div_diff(x, y)
        n = len(x)
        ax_bot.axis("tight"); ax_bot.axis("off")
        headers = ["x"] + [f"Δ^{j}" for j in range(n)]
        cell_data = []
        for i in range(n):
            row = [f"{x[i]:.4f}"]
            for j in range(n - i):
                row.append(f"{tab[i][j]:.4f}")
            for _ in range(i):
                row.append("")
            cell_data.append(row)
        tbl = ax_bot.table(cellText=cell_data, colLabels=headers, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        ax_bot.set_title("Table des différences divisées", pad=4)

        plt.tight_layout(); plt.show()

    # ── Moindres carrés ───────────────────────────────────────────────────
    def _ax3_moindres_carres(self):
        x, y = self._ax3_parse_points()
        if x is None: return
        try:
            d = int(self.ax3_deg_var.get())
        except:
            messagebox.showerror("Erreur", "Degré invalide"); return

        from approximation import poly_fit, eval_poly
        try:
            coeffs = poly_fit(x, y, d)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex)); return

        xx = np.linspace(x[0], x[-1], 400)
        yy = eval_poly(coeffs, xx)
        rmse = float(np.sqrt(np.mean((eval_poly(coeffs, x) - y)**2)))

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8),
                                              gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(f"Moindres Carrés — degré {d}")

        self._ax3_base_plot(ax_top, x, y, f"Ajustement polynomial (degré {d})")
        ax_top.plot(xx, yy, "r-", label=f"Gradient de degré {d}  (Erreur quadratique moyenne={rmse:.4f})")
        ax_top.legend()

        # table des coefficients
        ax_bot.axis("tight"); ax_bot.axis("off")
        cell_data = [[f"a_{i}", f"{c:.8f}"] for i, c in enumerate(coeffs)]
        tbl = ax_bot.table(cellText=cell_data,
                           colLabels=["Coefficient", "Valeur"],
                           loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        ax_bot.set_title(f"Coefficients  (Erreur quadratique moyenne = {rmse:.6f})", pad=4)

        plt.tight_layout(); plt.show()

    # ── Descente de gradient ──────────────────────────────────────────────
    def _ax3_gradient(self):
        x, y = self._ax3_parse_points()
        if x is None: return
        try:
            alpha = float(self.ax3_alpha_entry.get())
            d     = int(self.ax3_gd_deg_var.get())
        except:
            messagebox.showerror("Erreur", "Valeur de α ou degré invalide"); return

        # ── Normalisation des données (z-score sur x, min-max sur y) ──────
        # Cela rend la descente stable quelle que soit l'échelle des données.
        x_mean, x_std = float(np.mean(x)), float(np.std(x))
        y_mean, y_std = float(np.mean(y)), float(np.std(y))
        if x_std == 0: x_std = 1.0
        if y_std == 0: y_std = 1.0

        xn = (x - x_mean) / x_std   # x normalisé
        yn = (y - y_mean) / y_std    # y normalisé

        # Matrices de Vandermonde pour les x normalisés
        def poly_val(params, t):
            """Évalue le polynôme sur t (tableau numpy)."""
            val = np.zeros_like(t, dtype=float)
            for i, c in enumerate(params):
                val = val + c * t**i
            return val

        def loss(params):
            with np.errstate(over="raise", invalid="raise"):
                try:
                    return float(np.mean((poly_val(params, xn) - yn)**2))
                except FloatingPointError:
                    return float("inf")

        def grad_loss(params):
            err = poly_val(params, xn) - yn
            g = np.zeros(len(params))
            with np.errstate(over="raise", invalid="raise"):
                try:
                    for i in range(len(params)):
                        g[i] = 2.0 * float(np.mean(err * xn**i))
                except FloatingPointError:
                    g[:] = np.inf
            return g

        x0_params = np.zeros(d + 1)

        # Vérification préalable : si le gradient initial est déjà infini,
        # le pas est trop grand avant même de commencer.
        g0 = grad_loss(x0_params)
        if not np.all(np.isfinite(g0)):
            messagebox.showerror("Erreur",
                "Overflow dès la première itération.\n"
                "Essayez un pas α plus petit (ex. 0.001)."); return

        try:
            opt_params, hist_f, _ = descente_gradient_mod.descente_gradient(
                loss, grad_loss, x0_params, alpha=alpha, tol=1e-10, max_iter=10000)
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex)); return

        # Détecter la divergence
        if not np.all(np.isfinite(opt_params)) or not np.isfinite(hist_f[-1]):
            messagebox.showerror(
                "Divergence détectée",
                f"La descente a divergé avec α={alpha}.\n"
                "Essayez un pas plus petit (ex. α/10).")
            return

        # ── Dénormalisation pour l'affichage ──────────────────────────────
        # On évalue le polynôme dans l'espace normalisé puis on renormalise.
        xx_orig = np.linspace(x[0], x[-1], 400)
        xx_norm = (xx_orig - x_mean) / x_std
        yy_norm = poly_val(opt_params, xx_norm)
        yy_orig = yy_norm * y_std + y_mean          # retour à l'échelle originale

        # RMSE dans l'espace original
        y_pred_orig = poly_val(opt_params, xn) * y_std + y_mean
        rmse = float(np.sqrt(np.mean((y_pred_orig - y)**2)))

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f"Descente de Gradient — degré {d}, α={alpha}")

        # graphe du résultat
        self._ax3_base_plot(ax, x, y, f"Régression par gradient (degré {d})")
        ax.plot(xx_orig, yy_orig, "g-", label=f"Gradient de degré {d}  (Erreur quadratique moyenne={rmse:.4f})")
        ax.legend()

        plt.tight_layout(); plt.show()


# MAIN
if __name__ == "__main__":
    app = App()
    app.mainloop()
