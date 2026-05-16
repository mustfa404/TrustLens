import os
import sys
import threading
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils import resample

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

#  DATA PREPROCESSING
def preprocess(df, scaler=None, encoders=None, fit=True):
    df = df.copy()

    # Drop irrelevant columns
    drop_cols = ["Unnamed: 0", "trans_num", "first", "last",
                 "street", "trans_date_trans_time", "cc_num"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Feature engineering
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["age"] = (pd.Timestamp("2020-01-01") - df["dob"]).dt.days // 365
        df.drop(columns=["dob"], inplace=True)

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0]
                    if x in le.classes_ else -1
                )

    # Separate target
    y = None
    if "is_fraud" in df.columns:
        y = df.pop("is_fraud")

    # Scale
    if fit:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    else:
        X = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return X, y, scaler, encoders


def load_and_prepare(train_path, test_path, sample_size=50000, log_fn=None):
    def log(msg):
        if log_fn:
            log_fn(msg)

    log(f"Loading training data (up to {sample_size:,} rows)...")
    train_df = pd.read_csv(train_path, nrows=sample_size)
    log(f"  Train shape: {train_df.shape}  |  Fraud rate: {train_df['is_fraud'].mean()*100:.2f}%")

    log("Loading test data...")
    test_df = pd.read_csv(test_path)
    log(f"  Test shape:  {test_df.shape}  |  Fraud rate: {test_df['is_fraud'].mean()*100:.2f}%")

    log("Preprocessing (encoding, scaling)...")
    X_train_full, y_train_full, scaler, encoders = preprocess(train_df, fit=True)
    X_test, y_test, _, _ = preprocess(test_df, scaler=scaler, encoders=encoders, fit=False)

    log("Balancing training data (upsampling fraud class)...")
    df_combined = X_train_full.copy()
    df_combined["is_fraud"] = y_train_full.values
    majority = df_combined[df_combined["is_fraud"] == 0]
    minority = df_combined[df_combined["is_fraud"] == 1]
    minority_upsampled = resample(minority, replace=True,
                                  n_samples=len(majority) // 2, random_state=42)
    balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
    X_train = balanced.drop("is_fraud", axis=1)
    y_train = balanced["is_fraud"]
    log(f"  Balanced train: {X_train.shape}  |  Fraud rate: {y_train.mean()*100:.2f}%")

    return X_train, y_train, X_test, y_test, scaler, encoders


#  MODEL TRAINING & EVALUATION

MODELS = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12,
                                            n_jobs=-1, random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "SVM":           SVC(kernel="rbf", probability=True, random_state=42, cache_size=500),
}


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }


def train_all_models(X_train, y_train, X_test, y_test,
                     selected_models, mlflow_uri, experiment_name,
                     log_fn=None, progress_fn=None, result_fn=None):
    def log(msg):
        if log_fn: log_fn(msg)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    results = {}
    total = len(selected_models)

    for i, name in enumerate(selected_models):
        log(f"\n{'─'*40}")
        log(f"[{i+1}/{total}] Training: {name}")
        model = MODELS[name]

        with mlflow.start_run(run_name=name):
            # Log params
            params = model.get_params()
            mlflow.log_params({k: str(v) for k, v in list(params.items())[:15]})
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            t0 = datetime.now()
            model.fit(X_train, y_train)
            train_time = (datetime.now() - t0).total_seconds()
            log(f"  Training time: {train_time:.1f}s")

            metrics = evaluate(model, X_test, y_test)
            results[name] = metrics
            results[name]["train_time"] = train_time

            # Log metrics
            mlflow.log_metric("accuracy",  metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall",    metrics["recall"])
            mlflow.log_metric("f1",        metrics["f1"])
            mlflow.log_metric("roc_auc",   metrics["roc_auc"])
            mlflow.log_metric("train_time_sec", train_time)

            # Log model
            sig = infer_signature(X_train, model.predict(X_train[:5]))
            mlflow.sklearn.log_model(model, name.replace(" ", "_"),
                                     signature=sig,
                                     input_example=X_train.iloc[:3])

            log(f"  Accuracy:  {metrics['accuracy']:.4f}")
            log(f"  Precision: {metrics['precision']:.4f}")
            log(f"  Recall:    {metrics['recall']:.4f}")
            log(f"  F1-Score:  {metrics['f1']:.4f}")
            log(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            log(f"  MLflow run logged ✓")

        if progress_fn:
            progress_fn(int((i + 1) / total * 100))

    # Determine best model by F1
    best = max(results, key=lambda k: results[k]["f1"])
    log(f"\n{'═'*40}")
    log(f"✓ Best Model (by F1): {best}  |  F1={results[best]['f1']:.4f}")

    if result_fn:
        result_fn(results, best)

    return results, best


#  TKINTER GUI

COLORS = {
    "bg":        "#0F1117",
    "surface":   "#1A1D27",
    "card":      "#22263A",
    "accent":    "#6C63FF",
    "accent2":   "#FF6584",
    "success":   "#43D9AD",
    "warning":   "#FFB347",
    "text":      "#E8E8F0",
    "subtext":   "#9090AA",
    "border":    "#2E3250",
}

class FraudApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fraud Detection – MLflow Dashboard")
        self.geometry("1280x820")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        # State
        self.train_path = tk.StringVar()
        self.test_path  = tk.StringVar()
        self.sample_var = tk.IntVar(value=50000)
        self.experiment_name = tk.StringVar(value="FraudDetection")
        self.mlflow_uri = tk.StringVar(value="sqlite:///fraud_mlflow.db")
        self.model_vars = {n: tk.BooleanVar(value=True) for n in MODELS}
        self.results  = {}
        self.best_model = None
        self._data = None

        self._build_ui()

    # ── Layout ──────────────────────────────────────

    def _build_ui(self):
        # Top header
        hdr = tk.Frame(self, bg=COLORS["accent"], height=52)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🛡  Fraud Detection  ·  MLflow Classification Suite",
                 bg=COLORS["accent"], fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=20, pady=12)
        tk.Label(hdr, text="KNN · SVM · Random Forest · Decision Tree",
                 bg=COLORS["accent"], fg="#D0CFFF",
                 font=("Segoe UI", 10)).pack(side="right", padx=20)

        # Main paned window
        paned = tk.PanedWindow(self, orient="horizontal",
                               bg=COLORS["bg"], sashwidth=4,
                               sashrelief="flat", bd=0)
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        left  = self._build_left_panel(paned)
        right = self._build_right_panel(paned)
        paned.add(left,  minsize=300, width=320)
        paned.add(right, minsize=700)

    def _card(self, parent, title=None, **kw):
        frame = tk.Frame(parent, bg=COLORS["card"],
                         bd=0, highlightbackground=COLORS["border"],
                         highlightthickness=1)
        if title:
            tk.Label(frame, text=title, bg=COLORS["card"], fg=COLORS["accent"],
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
            ttk.Separator(frame).pack(fill="x", padx=8)
        return frame

    def _label(self, parent, text, size=9, bold=False, color=None):
        return tk.Label(parent, text=text,
                        bg=COLORS["card"],
                        fg=color or COLORS["text"],
                        font=("Segoe UI", size, "bold" if bold else "normal"))

    def _btn(self, parent, text, cmd, color=None, width=18):
        bg = color or COLORS["accent"]
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg="white", relief="flat",
                      font=("Segoe UI", 9, "bold"),
                      activebackground=COLORS["accent2"],
                      activeforeground="white",
                      cursor="hand2", width=width, pady=6)
        return b

    # ── Left panel ──────────────────────────────────

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS["bg"])

        # ─ Data files ─
        c1 = self._card(frame, "📂  Data Files")
        c1.pack(fill="x", padx=4, pady=4)

        for label, var, tag in [("Train CSV", self.train_path, "train"),
                                 ("Test CSV",  self.test_path,  "test")]:
            row = tk.Frame(c1, bg=COLORS["card"])
            row.pack(fill="x", padx=10, pady=4)
            self._label(row, label, size=9).pack(anchor="w")
            sub = tk.Frame(row, bg=COLORS["card"])
            sub.pack(fill="x")
            e = tk.Entry(sub, textvariable=var, bg=COLORS["surface"],
                         fg=COLORS["text"], insertbackground="white",
                         relief="flat", font=("Segoe UI", 8), bd=4)
            e.pack(side="left", fill="x", expand=True)
            self._btn(sub, "Browse", lambda t=tag: self._browse(t),
                      width=8).pack(side="right", padx=(4, 0))

        # Sample size
        row = tk.Frame(c1, bg=COLORS["card"])
        row.pack(fill="x", padx=10, pady=6)
        self._label(row, "Train sample size (rows):", size=9).pack(side="left")
        tk.Spinbox(row, from_=5000, to=500000, increment=5000,
                   textvariable=self.sample_var,
                   bg=COLORS["surface"], fg=COLORS["text"],
                   buttonbackground=COLORS["border"],
                   relief="flat", width=10,
                   font=("Segoe UI", 9)).pack(side="right")

        # ─ MLflow settings ─
        c2 = self._card(frame, "⚗️  MLflow Settings")
        c2.pack(fill="x", padx=4, pady=4)
        for label, var in [("Experiment name", self.experiment_name),
                            ("Tracking URI",   self.mlflow_uri)]:
            row = tk.Frame(c2, bg=COLORS["card"])
            row.pack(fill="x", padx=10, pady=4)
            self._label(row, label, size=9).pack(anchor="w")
            tk.Entry(row, textvariable=var,
                     bg=COLORS["surface"], fg=COLORS["text"],
                     insertbackground="white", relief="flat",
                     font=("Segoe UI", 8), bd=4).pack(fill="x")

        # ─ Model selection ─
        c3 = self._card(frame, "🤖  Models to Train")
        c3.pack(fill="x", padx=4, pady=4)
        for name, var in self.model_vars.items():
            row = tk.Frame(c3, bg=COLORS["card"])
            row.pack(fill="x", padx=12, pady=2)
            tk.Checkbutton(row, text=name, variable=var,
                           bg=COLORS["card"], fg=COLORS["text"],
                           selectcolor=COLORS["accent"],
                           activebackground=COLORS["card"],
                           activeforeground=COLORS["text"],
                           font=("Segoe UI", 9)).pack(side="left")

        # ─ Actions ─
        c4 = tk.Frame(frame, bg=COLORS["bg"])
        c4.pack(fill="x", padx=4, pady=4)
        self._btn(c4, "▶  Train All Models", self._run_training,
                  color=COLORS["accent"]).pack(fill="x", pady=3)
        self._btn(c4, "📊  Show Results", self._show_results,
                  color="#3A6186").pack(fill="x", pady=3)
        self._btn(c4, "🗑  Clear Log", self._clear_log,
                  color=COLORS["surface"]).pack(fill="x", pady=3)

        # Progress
        self.progress = ttk.Progressbar(frame, length=280, mode="determinate")
        self.progress.pack(padx=8, pady=6, fill="x")
        self.status_lbl = tk.Label(frame, text="Ready", bg=COLORS["bg"],
                                   fg=COLORS["subtext"],
                                   font=("Segoe UI", 8))
        self.status_lbl.pack()

        return frame

    # ── Right panel ─────────────────────────────────

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=COLORS["bg"])
        self.notebook = ttk.Notebook(frame)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",       background=COLORS["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",   background=COLORS["surface"],
                        foreground=COLORS["subtext"], padding=[14, 6],
                        font=("Segoe UI", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "white")])
        style.configure("TProgressbar",    troughcolor=COLORS["surface"],
                        background=COLORS["accent"], thickness=8)

        # Tab 1: Log
        self.log_frame = tk.Frame(self.notebook, bg=COLORS["bg"])
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, bg=COLORS["surface"], fg=COLORS["text"],
            font=("Consolas", 9), relief="flat", bd=0,
            insertbackground="white", wrap="word",
            selectbackground=COLORS["accent"])
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text.tag_config("green",  foreground=COLORS["success"])
        self.log_text.tag_config("yellow", foreground=COLORS["warning"])
        self.log_text.tag_config("purple", foreground=COLORS["accent"])
        self.log_text.tag_config("red",    foreground=COLORS["accent2"])

        # Tab 2: Metrics table
        self.metrics_frame = tk.Frame(self.notebook, bg=COLORS["bg"])

        # Tab 3: Charts
        self.chart_frame = tk.Frame(self.notebook, bg=COLORS["bg"])

        # Tab 4: Confusion matrices
        self.conf_frame = tk.Frame(self.notebook, bg=COLORS["bg"])

        self.notebook.add(self.log_frame,     text="📋  Training Log")
        self.notebook.add(self.metrics_frame, text="📈  Metrics")
        self.notebook.add(self.chart_frame,   text="📊  Charts")
        self.notebook.add(self.conf_frame,    text="🔢  Confusion Matrices")

        self.notebook.pack(fill="both", expand=True, padx=4, pady=4)
        frame.pack(fill="both", expand=True)
        return frame

    # ── Actions ─────────────────────────────────────

    def _browse(self, tag):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            if tag == "train": self.train_path.set(path)
            else:              self.test_path.set(path)

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    def _log(self, msg, tag=None):
        def _do():
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{ts}] {msg}\n", tag or "")
            self.log_text.see("end")
        self.after(0, _do)

    def _set_status(self, msg):
        self.after(0, lambda: self.status_lbl.config(text=msg))

    def _set_progress(self, val):
        self.after(0, lambda: self.progress.configure(value=val))

    def _run_training(self):
        if not self.train_path.get() or not self.test_path.get():
            messagebox.showerror("Missing Files",
                                 "Please select both Train and Test CSV files.")
            return
        selected = [n for n, v in self.model_vars.items() if v.get()]
        if not selected:
            messagebox.showerror("No Models", "Select at least one model.")
            return

        self._set_progress(0)
        self._set_status("Loading data…")
        self._log("═" * 50, "purple")
        self._log("Starting Fraud Detection Pipeline", "purple")
        self._log("═" * 50, "purple")

        def worker():
            try:
                X_tr, y_tr, X_te, y_te, scaler, enc = load_and_prepare(
                    self.train_path.get(), self.test_path.get(),
                    self.sample_var.get(), log_fn=self._log)
                self._data = (X_tr, y_tr, X_te, y_te)
                self._set_status("Training models…")

                results, best = train_all_models(
                    X_tr, y_tr, X_te, y_te,
                    selected_models=selected,
                    mlflow_uri=self.mlflow_uri.get(),
                    experiment_name=self.experiment_name.get(),
                    log_fn=self._log,
                    progress_fn=self._set_progress,
                    result_fn=self._on_results_ready,
                )
                self._set_status(f"Done ✓  Best: {best}")
                self._log("═" * 50, "green")
                self._log("Training complete! Click 'Show Results' for charts.", "green")
                self.after(0, lambda: self.notebook.select(1))

            except Exception as e:
                self._log(f"ERROR: {e}", "red")
                self._set_status("Error – see log")
                import traceback
                self._log(traceback.format_exc(), "red")

        threading.Thread(target=worker, daemon=True).start()

    def _on_results_ready(self, results, best):
        self.results = results
        self.best_model = best
        self.after(0, self._populate_metrics)
        self.after(0, self._draw_charts)
        self.after(0, self._draw_confusion_matrices)

    def _show_results(self):
        if not self.results:
            messagebox.showinfo("No Results", "Train models first.")
            return
        self.notebook.select(1)

    # ── Metrics Table ────────────────────────────────

    def _populate_metrics(self):
        for w in self.metrics_frame.winfo_children():
            w.destroy()

        tk.Label(self.metrics_frame, text="Model Evaluation Metrics",
                 bg=COLORS["bg"], fg=COLORS["accent"],
                 font=("Segoe UI", 13, "bold")).pack(pady=(12, 4))

        if self.best_model:
            tk.Label(self.metrics_frame,
                     text=f"🏆  Best Model (by F1): {self.best_model}",
                     bg=COLORS["bg"], fg=COLORS["success"],
                     font=("Segoe UI", 11, "bold")).pack(pady=2)

        # Table frame
        tbl = tk.Frame(self.metrics_frame, bg=COLORS["bg"])
        tbl.pack(fill="both", expand=True, padx=16, pady=8)

        headers = ["Model", "Accuracy", "Precision", "Recall",
                   "F1-Score", "ROC-AUC", "Train(s)"]
        col_w = [18, 10, 10, 10, 10, 10, 10]

        # Header row
        for j, (h, w) in enumerate(zip(headers, col_w)):
            tk.Label(tbl, text=h, bg=COLORS["accent"], fg="white",
                     font=("Segoe UI", 9, "bold"),
                     width=w, relief="flat", pady=7).grid(
                row=0, column=j, padx=1, pady=1, sticky="nsew")

        metrics_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "train_time"]
        for i, (name, res) in enumerate(self.results.items()):
            is_best = (name == self.best_model)
            bg = COLORS["card"] if i % 2 == 0 else COLORS["surface"]
            hi = COLORS["success"] if is_best else COLORS["text"]

            label = ("🏆 " if is_best else "   ") + name
            tk.Label(tbl, text=label, bg=bg, fg=hi,
                     font=("Segoe UI", 9, "bold" if is_best else "normal"),
                     width=18, anchor="w", padx=8, pady=6).grid(
                row=i+1, column=0, padx=1, pady=1, sticky="nsew")

            for j, key in enumerate(metrics_keys):
                val = res[key]
                txt = f"{val:.4f}" if key != "train_time" else f"{val:.1f}s"
                tk.Label(tbl, text=txt, bg=bg, fg=hi,
                         font=("Consolas", 9), width=col_w[j+1],
                         pady=6).grid(row=i+1, column=j+1, padx=1, pady=1, sticky="nsew")

        # Classification reports
        tk.Label(self.metrics_frame, text="Classification Reports",
                 bg=COLORS["bg"], fg=COLORS["accent"],
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=16, pady=(12, 2))

        rpt_nb = ttk.Notebook(self.metrics_frame)
        rpt_nb.pack(fill="both", expand=True, padx=12, pady=4)
        for name, res in self.results.items():
            f = tk.Frame(rpt_nb, bg=COLORS["surface"])
            t = scrolledtext.ScrolledText(f, bg=COLORS["surface"], fg=COLORS["text"],
                                          font=("Consolas", 9), relief="flat", height=10)
            t.insert("1.0", res["report"])
            t.config(state="disabled")
            t.pack(fill="both", expand=True, padx=4, pady=4)
            rpt_nb.add(f, text=name)

    # ── Charts ──────────────────────────────────────

    def _draw_charts(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        names   = list(self.results.keys())
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        colors  = ["#6C63FF", "#43D9AD", "#FFB347", "#FF6584", "#5BC0EB"]

        fig = plt.Figure(figsize=(11, 7), facecolor=COLORS["bg"])
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Bar charts per metric
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            vals = [self.results[n][metric] for n in names]
            bars = ax.bar(names, vals, color=color, alpha=0.85, width=0.5)
            ax.set_title(metric.upper().replace("_", " "), color=COLORS["text"],
                         fontsize=9, fontweight="bold")
            ax.set_ylim(0, 1.1)
            ax.set_facecolor(COLORS["surface"])
            ax.tick_params(colors=COLORS["subtext"], labelsize=7)
            ax.spines[:].set_color(COLORS["border"])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.3f}", ha="center", va="bottom",
                        color=COLORS["text"], fontsize=7.5, fontweight="bold")
            # Highlight best
            best_val = max(vals)
            for bar, val in zip(bars, vals):
                if val == best_val:
                    bar.set_edgecolor(COLORS["success"])
                    bar.set_linewidth(2)

        # Radar / summary table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(COLORS["surface"])
        ax6.spines[:].set_color(COLORS["border"])
        ax6.tick_params(colors=COLORS["subtext"], labelsize=7)
        ax6.set_title("F1  vs  ROC-AUC", color=COLORS["text"],
                      fontsize=9, fontweight="bold")
        for i, name in enumerate(names):
            x = self.results[name]["roc_auc"]
            y = self.results[name]["f1"]
            ax6.scatter(x, y, s=120, color=colors[i], zorder=5,
                        label=name, edgecolors="white", linewidth=1)
            ax6.annotate(name, (x, y), textcoords="offset points",
                         xytext=(6, 4), color=colors[i], fontsize=7.5)
        ax6.set_xlabel("ROC-AUC", color=COLORS["subtext"], fontsize=8)
        ax6.set_ylabel("F1-Score", color=COLORS["subtext"], fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # ── Confusion Matrices ───────────────────────────

    def _draw_confusion_matrices(self):
        for w in self.conf_frame.winfo_children():
            w.destroy()

        n = len(self.results)
        cols = min(n, 2)
        rows = (n + cols - 1) // cols

        fig = plt.Figure(figsize=(5.5 * cols, 4.5 * rows),
                         facecolor=COLORS["bg"])

        cmap = plt.cm.Blues

        for idx, (name, res) in enumerate(self.results.items()):
            ax = fig.add_subplot(rows, cols, idx + 1)
            cm = res["conf_matrix"]
            im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.04)

            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Legit", "Fraud"], color=COLORS["text"], fontsize=9)
            ax.set_yticklabels(["Legit", "Fraud"], color=COLORS["text"], fontsize=9)
            ax.set_xlabel("Predicted", color=COLORS["subtext"])
            ax.set_ylabel("Actual",    color=COLORS["subtext"])

            title_color = COLORS["success"] if name == self.best_model else COLORS["text"]
            prefix = "🏆 " if name == self.best_model else ""
            ax.set_title(f"{prefix}{name}", color=title_color,
                         fontsize=10, fontweight="bold")
            ax.set_facecolor(COLORS["surface"])
            ax.spines[:].set_color(COLORS["border"])
            ax.tick_params(colors=COLORS["subtext"])

            total = cm.sum()
            for r in range(2):
                for c in range(2):
                    val = cm[r, c]
                    pct = val / total * 100
                    color = "white" if cm[r, c] > cm.max() / 2 else "black"
                    ax.text(c, r, f"{val:,}\n({pct:.1f}%)",
                            ha="center", va="center",
                            color=color, fontsize=9, fontweight="bold")

        fig.tight_layout(pad=2.5)
        canvas = FigureCanvasTkAgg(fig, master=self.conf_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)


# ──────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    app = FraudApp()
    app.mainloop()