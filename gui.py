"""Japanese desktop interface. Run: uv run python gui.py."""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from evaluate_dataset import build_evaluator, evaluate_system
from report_evaluator.dataset_io import load_dataset_records
from report_evaluator.gui_helpers import format_metrics, make_request
from report_evaluator.schemas import EvaluationRequest

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError as exc:
    raise SystemExit(
        "GUIにはTk対応のPythonが必要です。READMEのGUIセットアップを確認してください。\n"
        + str(exc)
    ) from exc


class ReportEvaluatorApp:
    def __init__(self, root):
        self.root = root
        root.title("レポート評価 | Report Evaluator")
        root.geometry("1180x840")
        root.minsize(860, 650)
        self.queue = Queue()
        self.busy = False
        self.result = None
        self.metrics = None
        self.snapshot = None
        self.status = tk.StringVar(
            value="レポートを貼り付けるか、ファイルを開いてください。"
        )
        self.system = tk.StringVar(value="baseline")
        self.actions = []
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6fa")
        style.configure(
            "TLabel", background="#f4f6fa", foreground="#243047", font=("", 11)
        )
        style.configure("TButton", padding=(12, 7))
        style.configure("Title.TLabel", font=("", 22, "bold"))
        style.configure("Score.TLabel", font=("", 14, "bold"))
        shell = ttk.Frame(root, padding=20)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="レポート校正システム", style="Title.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            shell, text="冗長性・論理整合性・要求網羅度を確認し、推敲に役立てます。"
        ).pack(anchor="w", pady=(5, 12))
        bar = ttk.Frame(shell)
        bar.pack(fill="x", pady=(0, 10))
        ttk.Label(bar, text="評価方式").pack(side="left")
        self.selector = ttk.Combobox(
            bar,
            textvariable=self.system,
            values=("baseline", "improved"),
            state="readonly",
            width=12,
        )
        self.selector.pack(side="left", padx=8)
        ttk.Label(bar, text="baseline: メモの手法 / improved: 追加モデルが必要").pack(
            side="left"
        )
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        self.input_tab = ttk.Frame(self.tabs, padding=12)
        self.output_tab = ttk.Frame(self.tabs, padding=12)
        self.metrics_tab = ttk.Frame(self.tabs, padding=12)
        for frame, title in [
            (self.input_tab, "1  入力"),
            (self.output_tab, "2  評価結果"),
            (self.metrics_tab, "3  システム評価"),
        ]:
            self.tabs.add(frame, text=title)
        self.build_input()
        self.build_results()
        self.build_metrics()
        ttk.Label(shell, textvariable=self.status, wraplength=1000).pack(
            anchor="w", pady=(10, 0)
        )
        root.after(100, self.poll)

    def button(self, parent, title, action):
        button = ttk.Button(parent, text=title, command=action)
        button.pack(side="left", padx=(0, 8))
        self.actions.append(button)
        return button

    def text_box(self, parent, height=5, readonly=False):
        widget = ScrolledText(
            parent,
            height=height,
            wrap="word",
            font=("", 12),
            padx=12,
            pady=10,
            relief="flat",
            borderwidth=1,
            undo=not readonly,
        )
        if readonly:
            widget.configure(state="disabled")
        return widget

    def build_input(self):
        buttons = ttk.Frame(self.input_tab)
        buttons.pack(fill="x", pady=(0, 10))
        self.button(buttons, "本文を開く (.txt)", self.open_text)
        self.button(buttons, "サンプルを入力", self.load_sample)
        self.button(buttons, "評価する", self.evaluate)
        ttk.Label(self.input_tab, text="レポート本文").pack(anchor="w")
        self.report = self.text_box(self.input_tab, 11)
        self.report.pack(fill="both", expand=True, pady=(4, 10))
        ttk.Label(
            self.input_tab, text="問題文（baselineでは採点に直接使用しません）"
        ).pack(anchor="w")
        self.prompt = self.text_box(self.input_tab, 3)
        self.prompt.pack(fill="x", pady=(4, 10))
        words = ttk.Frame(self.input_tab)
        words.pack(fill="x")
        words.columnconfigure((0, 1), weight=1)
        ttk.Label(words, text="必須キーワード（カンマ・読点・改行で区切る）").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(words, text="推奨キーワード（任意）").grid(
            row=0, column=1, sticky="w"
        )
        self.required = self.text_box(words, 2)
        self.recommended = self.text_box(words, 2)
        self.required.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=4)
        self.recommended.grid(row=1, column=1, sticky="ew", pady=4)

    def build_results(self):
        self.score = tk.StringVar(
            value="まだ評価結果がありません。入力画面で「評価する」を押してください。"
        )
        ttk.Label(
            self.output_tab,
            textvariable=self.score,
            style="Score.TLabel",
            wraplength=980,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            self.output_tab,
            text="点数は高いほど良好。自動検出は候補であり、文章の正しさを保証しません。結果は評価時の本文です。",
            wraplength=980,
        ).pack(anchor="w")
        bar = ttk.Frame(self.output_tab)
        bar.pack(fill="x", pady=8)
        self.save_result = self.button(
            bar,
            "入力・結果をJSONで保存",
            lambda: self.save(self.result, "report_result.json"),
        )
        notebook = ttk.Notebook(self.output_tab)
        notebook.pack(fill="both", expand=True)
        self.views = {}
        for key, label, legend in [
            (
                "redundancy",
                "冗長性",
                "黄色マーカー：似た文の繰り返し。新しい情報がなければ文を統合・削除しましょう。",
            ),
            (
                "contradiction",
                "論理整合性",
                "赤い下線：矛盾候補。前後の文の主語・条件・否定表現を確認しましょう。",
            ),
            (
                "keyword",
                "要求網羅度",
                "緑色マーカー：一致したキーワード。未検出の必須語は説明や根拠を補いましょう。",
            ),
        ]:
            frame = ttk.Frame(notebook, padding=10)
            notebook.add(frame, text=label)
            ttk.Label(frame, text=legend, wraplength=950).pack(anchor="w", pady=(0, 8))
            view = self.text_box(frame, 10, True)
            view.pack(fill="both", expand=True)
            view.tag_configure("redundancy", background="#fff0a6", foreground="#493700")
            view.tag_configure("contradiction", foreground="#b42332", underline=True)
            view.tag_configure("keyword", background="#c9f1df", foreground="#114f3b")
            details = self.text_box(frame, 7, True)
            details.pack(fill="both", expand=True, pady=(10, 0))
            self.views[key] = (view, details)
        self.save_result.configure(state="disabled")

    def build_metrics(self):
        ttk.Label(
            self.metrics_tab,
            text="正解データと比較して、評価システムの性能を確認",
            style="Score.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.metrics_tab,
            text="人手の正解ラベルを含むJSON / JSONLを選択してください。単一レポートの点数とは別の指標です。",
            wraplength=980,
        ).pack(anchor="w", pady=8)
        bar = ttk.Frame(self.metrics_tab)
        bar.pack(fill="x", pady=(0, 10))
        self.button(bar, "正解データを選んで評価", self.evaluate_dataset)
        self.save_metrics = self.button(
            bar,
            "指標・誤り分析をJSONで保存",
            lambda: self.save(self.metrics, "system_evaluation.json"),
        )
        self.metrics_view = self.text_box(self.metrics_tab, 20, True)
        self.metrics_view.pack(fill="both", expand=True)
        self.save_metrics.configure(state="disabled")

    @staticmethod
    def replace(widget, text):
        previous = str(widget.cget("state"))
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state=previous)

    def open_text(self):
        path = filedialog.askopenfilename(filetypes=[("UTF-8テキスト", "*.txt")])
        if path:
            try:
                self.replace(self.report, Path(path).read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError) as exc:
                messagebox.showerror("読み込みに失敗しました", str(exc))

    def load_sample(self):
        try:
            req = EvaluationRequest.model_validate_json(
                (Path(__file__).parent / "examples/input.json").read_text(
                    encoding="utf-8"
                )
            )
            for widget, value in [
                (self.report, req.report.text),
                (self.prompt, req.assignment.prompt),
                (self.required, "\n".join(req.assignment.required_keywords)),
                (self.recommended, "\n".join(req.assignment.recommended_keywords)),
            ]:
                self.replace(widget, value)
        except (OSError, ValueError) as exc:
            messagebox.showerror("サンプルを開けません", str(exc))

    def start_job(self, job, callback):
        if self.busy:
            return
        self.busy = True
        for button in self.actions:
            button.configure(state="disabled")
        self.selector.configure(state="disabled")
        self.status.set(
            "評価中です。improvedの初回はモデルの読み込みに時間がかかります。"
        )

        def work():
            try:
                self.queue.put((callback, job(), None))
            except Exception as exc:
                self.queue.put((callback, None, str(exc)))

        Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            callback, value, error = self.queue.get_nowait()
        except Empty:
            pass
        else:
            self.busy = False
            for button in self.actions:
                button.configure(state="normal")
            self.selector.configure(state="readonly")
            if error is not None:
                self.status.set(
                    "評価に失敗しました。入力や依存関係を確認して再実行してください。"
                )
                messagebox.showerror(
                    "評価に失敗しました",
                    error
                    + "\n\nimprovedには uv sync --extra improved と初回のモデル取得が必要です。",
                )
            else:
                callback(value)
                self.status.set(
                    "評価が完了しました。入力を編集した場合は、もう一度評価してください。"
                )
            self.save_result.configure(state="normal" if self.result else "disabled")
            self.save_metrics.configure(state="normal" if self.metrics else "disabled")
        self.root.after(100, self.poll)

    def evaluate(self):
        try:
            req = make_request(
                *(
                    w.get("1.0", "end-1c")
                    for w in (self.report, self.prompt, self.required, self.recommended)
                )
            )
        except ValueError as exc:
            messagebox.showerror("入力を確認してください", str(exc))
            return
        system = self.system.get()
        self.start_job(
            lambda: (req, build_evaluator(system).evaluate(req)), self.show_results
        )

    def show_results(self, value):
        req, result = value
        self.snapshot = req
        self.result = {"request": req.model_dump(), "result": result.model_dump()}
        coverage = (
            f"{result.coverage_score:.1f}"
            if result.keywords
            else "未評価（キーワード未指定）"
        )
        self.score.set(
            f"{result.system}  |  総合 {result.overall_score_100:.1f}/100  |  冗長性 {result.redundancy_score:.1f}  |  論理整合性 {result.consistency_score:.1f}  |  網羅度 {coverage}"
        )
        for key, (view, detail) in self.views.items():
            self.replace(view, req.report.text)
            highlights = [h for h in result.highlights if h.type == key]
            for h in highlights:
                # Tcl counts astral characters as two units; derive offsets using its own string length.
                start = self.root.tk.call(
                    "string", "length", req.report.text[: h.start]
                )
                end = self.root.tk.call("string", "length", req.report.text[: h.end])
                view.tag_add(key, f"1.0 + {start} chars", f"1.0 + {end} chars")
            lines = []
            if key == "keyword":
                for k in result.keywords:
                    lines.append(
                        f"{'必須' if k.required else '推奨'} / {k.keyword} : {'検出（' + k.match_type + '）' if k.matched else '未検出 → 本文に説明を追加してください'}"
                    )
                if not result.keywords:
                    lines.append(
                        "キーワード未指定のため網羅度は未評価です。既存処理では100点として総合点に含まれます。"
                    )
                lines.append(
                    "キーワードの出現は、問題文への十分な回答を意味しません。baselineは空白除去・大小文字統一後に照合します。表記によってはマーカーが付かない場合があります。"
                )
            else:
                for h in highlights:
                    lines.append(
                        f"文{h.sentence_index + 1 if h.sentence_index is not None else '?'}：{h.reason}\n「{req.report.text[h.start : h.end]}」"
                    )
                if not highlights:
                    lines.append("該当する指摘箇所は検出されませんでした。")
                if key == "redundancy":
                    d = result.diagnostics.get("redundancy", {})
                    lines.append(
                        f"bigram反復率: {d.get('bigram_repeat_ratio', '—')} / trigram反復率: {d.get('trigram_repeat_ratio', '—')}"
                    )
                    lines.append(
                        "反復率は本文全体の指標、マーカーは類似文の指摘です。重複文は要点を残して統合してください。"
                    )
                else:
                    lines.append(
                        "同じ対象について肯定と否定が混在していないか確認し、条件や例外がある場合は明記してください。"
                    )
            self.replace(detail, "\n\n".join(lines))
        self.tabs.select(self.output_tab)

    def evaluate_dataset(self):
        path = filedialog.askopenfilename(filetypes=[("正解データ", "*.json *.jsonl")])
        if not path:
            return
        system = self.system.get()

        def job():
            records = load_dataset_records(path)
            if not records:
                raise ValueError("正解データが空です。")
            return evaluate_system(system, records)

        self.start_job(job, self.show_metrics)

    def show_metrics(self, value):
        self.metrics = value
        self.replace(self.metrics_view, format_metrics(value))

    def save(self, payload, filename):
        if payload is None:
            return
        path = filedialog.asksaveasfilename(
            initialfile=filename,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            try:
                Path(path).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                self.status.set("保存しました。")
            except OSError as exc:
                messagebox.showerror("保存に失敗しました", str(exc))


def main():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(
            "GUIの起動にはデスクトップ環境とTkが必要です。\n" + str(exc)
        ) from exc
    ReportEvaluatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
