from __future__ import annotations

"""Export history mixin for AppHost — CSV/XLSX export dialog and progress UI."""

import queue
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from services.history_export_coordinator import HistoryExportCoordinator
from services.history_result_export_service import HistoryExportEntry, HistoryResultExportService


class HostExportMixin:
    """Export history dialog mixin."""

    def export_history_results(self):
        return self._export_history_results()

    def _get_history_export_coordinator(self) -> HistoryExportCoordinator:
        coordinator = self.__dict__.get("_history_export_coordinator", None)
        if not isinstance(coordinator, HistoryExportCoordinator):
            coordinator = HistoryExportCoordinator()
            self._history_export_coordinator = coordinator
        return coordinator

    def _export_history_results(self) -> None:
        try:
            service = self._make_history_export_service()
        except Exception:
            messagebox.showerror("导出错误", "无法创建导出服务")
            return
        entries = service.list_exportable_entries()
        if not entries:
            messagebox.showinfo("提示", "当前没有可导出的历史结果")
            return
        self._show_history_export_dialog(entries=entries, service=service)

    def _show_history_export_dialog(
        self,
        *,
        entries: list[HistoryExportEntry],
        service: HistoryResultExportService,
    ) -> None:
        selected_entries: dict[str, list[HistoryExportEntry] | None] = {"value": None}
        top = tk.Toplevel(self)
        top.title("导出历史结果")
        top.transient(self)
        try:
            top.grab_set()
        except Exception:
            pass

        frm = ttk.Frame(top, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="选择需要导出的完整测量记录。").pack(fill=tk.X, pady=(0, 8))

        tree_wrap = ttk.Frame(frm)
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(
            tree_wrap,
            columns=("recipe_name", "serial", "run_id", "status", "date"),
            show="tree headings",
            height=18,
        )
        tree.column("#0", width=30, stretch=False)
        tree.heading("#0", text="")
        tree.column("recipe_name", width=160, stretch=True)
        tree.heading("recipe_name", text="配方")
        tree.column("serial", width=140, stretch=True)
        tree.heading("serial", text="管号")
        tree.column("run_id", width=100, stretch=True)
        tree.heading("run_id", text="运行ID")
        tree.column("status", width=70, stretch=False)
        tree.heading("status", text="状态")
        tree.column("date", width=100, stretch=False)
        tree.heading("date", text="日期")
        ysb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

        entry_by_iid: dict[str, HistoryExportEntry] = {}
        recipe_children: dict[str, list[str]] = {}
        date_sort_order: list[bool] = [True]

        def _toggle_date_sort() -> None:
            date_sort_order[0] = not date_sort_order[0]
            _render_tree()

        def _select_all() -> None:
            for iid in entry_by_iid:
                try:
                    tree.selection_add(iid)
                except Exception:
                    pass

        def _clear_selection() -> None:
            for iid in list(tree.selection()):
                try:
                    tree.selection_remove(iid)
                except Exception:
                    pass

        def _confirm() -> None:
            iids = list(tree.selection())
            selected: list[HistoryExportEntry] = []
            for iid in iids:
                entry = entry_by_iid.get(iid)
                if entry is not None:
                    selected.append(entry)
                child_iids = recipe_children.get(iid, [])
                for ciid in child_iids:
                    centry = entry_by_iid.get(ciid)
                    if centry is not None and centry not in selected:
                        selected.append(centry)
            if not selected:
                messagebox.showwarning("提示", "未选择任何记录")
                return
            selected_entries["value"] = selected
            try:
                top.grab_release()
            except Exception:
                pass
            try:
                top.destroy()
            except Exception:
                pass

        def _cancel() -> None:
            selected_entries["value"] = []
            try:
                top.grab_release()
            except Exception:
                pass
            try:
                top.destroy()
            except Exception:
                pass

        top.protocol("WM_DELETE_WINDOW", _cancel)

        def _render_tree() -> None:
            for item in tree.get_children():
                tree.delete(item)
            entry_by_iid.clear()
            recipe_children.clear()
            recipe_order: list[str] = []
            for entry in entries:
                recipe_name = entry.recipe_name or ""
                if recipe_name not in recipe_order:
                    recipe_order.append(recipe_name)
            if date_sort_order[0]:
                recipe_order.sort(key=lambda rn: max(
                    (e.date or "" for e in entries if (e.recipe_name or "") == rn),
                    default="",
                ), reverse=True)
            else:
                recipe_order.sort(key=lambda rn: min(
                    (e.date or "" for e in entries if (e.recipe_name or "") == rn),
                    default="",
                ))
            for recipe_name in recipe_order:
                group = [e for e in entries if (e.recipe_name or "") == recipe_name]
                if date_sort_order[0]:
                    group.sort(key=lambda e: e.date or "", reverse=True)
                else:
                    group.sort(key=lambda e: e.date or "")
                parent_iid = f"recipe_{recipe_name}"
                tree.insert(
                    "",
                    tk.END,
                    iid=parent_iid,
                    text=f"[ ] {recipe_name}",
                    values=("", "", "", "", ""),
                    open=True,
                )
                recipe_children[parent_iid] = []
                for entry in group:
                    status_text = str(entry.status or "")
                    date_text = str(entry.date or "")
                    serial_text = str(entry.serial or "")
                    run_id_text = str(entry.run_id or "")
                    child_iid = f"run_{entry.serial}_{entry.run_id}"
                    tree.insert(
                        parent_iid,
                        tk.END,
                        iid=child_iid,
                        text="",
                        values=(recipe_name, serial_text, run_id_text, status_text, date_text),
                    )
                    entry_by_iid[child_iid] = entry
                    recipe_children[parent_iid].append(child_iid)
                if parent_iid not in entry_by_iid:
                    entry_by_iid[parent_iid] = HistoryExportEntry(
                        serial="",
                        run_id="",
                        recipe_name=recipe_name,
                        status="",
                        date="",
                        data={},
                    )
                    recipe_children[parent_iid] = recipe_children.get(parent_iid, [])

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X, pady=(10, 0))
        sort_button = ttk.Button(btn_row, text="日期倒序", command=_toggle_date_sort)
        sort_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="全选", command=_select_all).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="清空", command=_clear_selection).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="导出", command=_confirm).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btn_row, text="取消", command=_cancel).pack(side=tk.RIGHT)

        _render_tree()

        try:
            top.wait_window()
        except Exception:
            pass

        chosen = selected_entries.get("value")
        if not chosen:
            return

        self._start_history_export_with_progress(entries=chosen, service=service)

    def _start_history_export_with_progress(
        self,
        *,
        entries: list[HistoryExportEntry],
        service: HistoryResultExportService,
    ) -> None:
        progress = self._show_history_export_progress()
        result_q: queue.Queue = queue.Queue()

        def _run() -> None:
            try:
                for entry in entries:
                    service.export_entry(entry)
                result_q.put(("done", "导出完成"))
            except Exception as exc:
                result_q.put(("error", str(exc)))

        import threading
        threading.Thread(target=_run, daemon=True).start()

        def _poll_result() -> None:
            try:
                status, payload = result_q.get_nowait()
            except queue.Empty:
                try:
                    self.after(100, _poll_result)
                except Exception:
                    pass
                return
            try:
                if progress.winfo_exists():
                    progress.destroy()
            except Exception:
                pass
            if status == "error":
                messagebox.showerror("导出错误", str(payload))
            self.auto_msg_var.set(
                str(self.auto_msg_var.get() or "") + f" | 导出完成: {self._compact_status_path(payload)}"
            )

        try:
            self.after(100, _poll_result)
        except Exception:
            pass

    def _show_history_export_progress(self) -> tk.Toplevel:
        top = tk.Toplevel(self)
        top.title("导出结果")
        top.transient(self)
        try:
            top.grab_set()
        except Exception:
            pass
        frm = ttk.Frame(top, padding=18)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="导出中，请等待...", font=("Segoe UI", 10, "bold")).pack(fill=tk.X)
        ttk.Label(frm, text="当前导出过程不可中断。").pack(fill=tk.X, pady=(8, 0))
        try:
            top.protocol("WM_DELETE_WINDOW", lambda: None)
        except Exception:
            pass
        return top
