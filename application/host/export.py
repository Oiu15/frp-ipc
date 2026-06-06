from __future__ import annotations

"""Export history mixin for AppHost - CSV/XLSX export dialog and progress UI."""

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, cast

from services.history_export_coordinator import HistoryExportCoordinator
from services.history_result_export_service import HistoryExportEntry, HistoryResultExportService


class HostExportMixin:
    """Mixin providing the manual history export workflow."""

    _history_export_coordinator: HistoryExportCoordinator

    if TYPE_CHECKING:
        def _make_history_export_service(self) -> HistoryResultExportService: ...

    def export_history_results(self):
        return self._export_history_results()

    def _get_history_export_coordinator(self) -> HistoryExportCoordinator:
        coordinator = self.__dict__.get("_history_export_coordinator", None)
        if not isinstance(coordinator, HistoryExportCoordinator):
            coordinator = HistoryExportCoordinator()
            self._history_export_coordinator = coordinator
        return coordinator

    def _export_history_results(self) -> None:
        host = cast(tk.Tk, self)
        try:
            coordinator = self._get_history_export_coordinator()
            service = self._make_history_export_service()
            entries = coordinator.list_exportable_entries(service)
        except Exception as exc:
            try:
                messagebox.showerror("导出结果", f"读取历史结果失败：{exc}", parent=host)
            except Exception:
                pass
            return None

        if not entries:
            try:
                messagebox.showinfo("导出结果", "没有找到可导出的完整测量历史。", parent=host)
            except Exception:
                pass
            return None

        self._show_history_export_dialog(entries, service)
        return None

    def _show_history_export_dialog(
        self,
        entries: list[HistoryExportEntry],
        service: HistoryResultExportService,
    ) -> None:
        host = cast(tk.Tk, self)
        selected_entries: dict[str, list[HistoryExportEntry] | None] = {"value": None}
        top = tk.Toplevel(host)
        top.title("导出历史结果")
        top.transient(host)
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
            columns=("start_time", "serial", "recipe_name", "status"),
            show="tree headings",
            selectmode="none",
            height=16,
        )
        tree.heading("#0", text="选择 / 日期")
        tree.heading("start_time", text="开始时间")
        tree.heading("serial", text="流水号")
        tree.heading("recipe_name", text="配方")
        tree.heading("status", text="状态")
        tree.column("#0", width=110, stretch=False)
        tree.column("start_time", width=160, stretch=False)
        tree.column("serial", width=210, stretch=True)
        tree.column("recipe_name", width=160, stretch=True)
        tree.column("status", width=70, stretch=False)
        ysb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

        entry_by_iid: dict[str, HistoryExportEntry] = {}
        child_iids: list[str] = []
        date_parent: dict[str, str] = {}
        date_label: dict[str, str] = {}
        date_children: dict[str, list[str]] = {}
        selected_keys: set[tuple[str, str, str]] = set()
        date_desc_state = {"value": True}
        sort_button: ttk.Button | None = None

        def _entry_key(entry: HistoryExportEntry) -> tuple[str, str, str]:
            return (str(entry.date), str(entry.serial), str(entry.run_id))

        def _sort_entries_for_dialog() -> list[HistoryExportEntry]:
            if date_desc_state["value"]:
                return sorted(entries, key=lambda item: (item.date, item.sort_ts, item.serial), reverse=True)
            return sorted(entries, key=lambda item: (item.date, item.sort_ts, item.serial))

        def _checked_text(checked: bool) -> str:
            return "[x]" if checked else "[ ]"

        def _parent_text(parent_iid: str) -> str:
            label = date_label.get(parent_iid, parent_iid)
            children = date_children.get(parent_iid, [])
            selected_count = sum(1 for iid in children if _entry_key(entry_by_iid[iid]) in selected_keys)
            if selected_count <= 0:
                marker = "[ ]"
            elif selected_count >= len(children):
                marker = "[x]"
            else:
                marker = "[-]"
            return f"{marker} {label}"

        def _refresh_checkmarks() -> None:
            for iid in child_iids:
                try:
                    checked = _entry_key(entry_by_iid[iid]) in selected_keys
                    tree.item(iid, text=f"{_checked_text(checked)}")
                except Exception:
                    pass
            for parent_iid in date_children:
                try:
                    tree.item(parent_iid, text=_parent_text(parent_iid))
                except Exception:
                    pass

        def _set_parent_checked(parent_iid: str, checked: bool) -> None:
            for iid in date_children.get(parent_iid, []):
                key = _entry_key(entry_by_iid[iid])
                if checked:
                    selected_keys.add(key)
                else:
                    selected_keys.discard(key)
            _refresh_checkmarks()

        def _toggle_iid(iid: str) -> None:
            if iid in entry_by_iid:
                key = _entry_key(entry_by_iid[iid])
                if key in selected_keys:
                    selected_keys.discard(key)
                else:
                    selected_keys.add(key)
                _refresh_checkmarks()
                return
            if iid in date_children:
                children = date_children.get(iid, [])
                should_check = any(_entry_key(entry_by_iid[child]) not in selected_keys for child in children)
                _set_parent_checked(iid, should_check)

        def _render_tree() -> None:
            open_dates: set[str] = set()
            for parent_iid, date_text in date_label.items():
                try:
                    if bool(tree.item(parent_iid, "open")):
                        open_dates.add(str(date_text))
                except Exception:
                    pass
            if not date_label:
                open_dates = {entry.date for entry in entries}

            try:
                tree.delete(*tree.get_children())
            except Exception:
                pass
            entry_by_iid.clear()
            child_iids.clear()
            date_parent.clear()
            date_label.clear()
            date_children.clear()

            for index, entry in enumerate(_sort_entries_for_dialog()):
                parent_iid = date_parent.get(entry.date)
                if parent_iid is None:
                    parent_iid = f"date:{entry.date}"
                    date_parent[entry.date] = parent_iid
                    date_label[parent_iid] = entry.date
                    date_children[parent_iid] = []
                    tree.insert(
                        "",
                        tk.END,
                        iid=parent_iid,
                        text=f"[ ] {entry.date}",
                        open=(entry.date in open_dates),
                    )
                iid = f"run:{index}"
                entry_by_iid[iid] = entry
                child_iids.append(iid)
                date_children.setdefault(parent_iid, []).append(iid)
                tree.insert(
                    parent_iid,
                    tk.END,
                    iid=iid,
                    text="[ ]",
                    values=(entry.start_time, entry.serial, entry.recipe_name, entry.status),
                )
            _refresh_checkmarks()

        def _refresh_sort_button() -> None:
            if sort_button is None:
                return
            try:
                sort_button.configure(text=("日期倒序" if date_desc_state["value"] else "日期正序"))
            except Exception:
                pass

        def _toggle_date_sort() -> None:
            date_desc_state["value"] = not date_desc_state["value"]
            _render_tree()
            _refresh_sort_button()

        def _handle_tree_click(event) -> str | None:
            try:
                region = str(tree.identify("region", int(event.x), int(event.y)))
                column = str(tree.identify_column(int(event.x)))
                iid = str(tree.identify_row(int(event.y)) or "")
                element = str(tree.identify("element", int(event.x), int(event.y)) or "")
            except Exception:
                return None
            if not iid:
                return None
            if iid in date_children and "indicator" in element.lower():
                return None
            if region in {"tree", "cell"} and column == "#0":
                _toggle_iid(iid)
                return "break"
            return None

        def _toggle_focused(_event=None) -> str:
            try:
                iid = str(tree.focus() or "")
            except Exception:
                iid = ""
            if not iid:
                try:
                    selected = tree.selection()
                    iid = str(selected[0]) if selected else ""
                except Exception:
                    iid = ""
            if iid:
                _toggle_iid(iid)
            return "break"

        try:
            tree.bind("<Button-1>", _handle_tree_click)
            tree.bind("<space>", _toggle_focused)
        except Exception:
            pass

        def _select_all() -> None:
            selected_keys.update(_entry_key(entry_by_iid[iid]) for iid in child_iids if iid in entry_by_iid)
            _refresh_checkmarks()

        def _clear_selection() -> None:
            selected_keys.clear()
            _refresh_checkmarks()

        def _confirm() -> None:
            selected = [
                entry_by_iid[iid]
                for iid in child_iids
                if iid in entry_by_iid and _entry_key(entry_by_iid[iid]) in selected_keys
            ]
            if not selected:
                messagebox.showwarning("导出结果", "请先选择至少一条测量记录。", parent=top)
                return
            selected_entries["value"] = selected
            try:
                top.destroy()
            except Exception:
                pass

        def _cancel() -> None:
            selected_entries["value"] = None
            try:
                top.destroy()
            except Exception:
                pass

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X, pady=(10, 0))
        sort_button = ttk.Button(btn_row, text="日期倒序", command=_toggle_date_sort)
        sort_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="全选", command=_select_all).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="清空", command=_clear_selection).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="导出", command=_confirm).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btn_row, text="取消", command=_cancel).pack(side=tk.RIGHT)

        _render_tree()
        _refresh_sort_button()
        top.protocol("WM_DELETE_WINDOW", _cancel)
        top.bind("<Return>", lambda _e: _confirm())
        top.bind("<Escape>", lambda _e: _cancel())
        try:
            top.wait_window()
        except Exception:
            return None

        selected = selected_entries.get("value")
        if not selected:
            return None

        try:
            save_path = filedialog.asksaveasfilename(
                parent=host,
                title="导出检测数据汇总",
                initialfile="检测数据汇总.xlsx",
                defaultextension=".xlsx",
                filetypes=[("Excel 工作簿", "*.xlsx"), ("所有文件", "*.*")],
            )
        except Exception as exc:
            try:
                messagebox.showerror("导出结果", f"选择导出文件失败：{exc}", parent=host)
            except Exception:
                pass
            return None
        if not save_path:
            return None

        self._start_history_export_with_progress(service, list(selected), Path(save_path))
        return None

    def _start_history_export_with_progress(
        self,
        service: HistoryResultExportService,
        entries: list[HistoryExportEntry],
        output_path: Path,
    ) -> None:
        host = cast(tk.Tk, self)
        progress = self._show_history_export_progress()
        result_q = self._get_history_export_coordinator().start_export(service, entries, output_path)

        def _close_progress() -> None:
            try:
                progress.grab_release()
            except Exception:
                pass
            try:
                progress.destroy()
            except Exception:
                pass

        def _poll_result() -> None:
            try:
                status, payload = result_q.get_nowait()
            except queue.Empty:
                try:
                    host.after(100, _poll_result)
                except Exception:
                    pass
                return

            _close_progress()
            if status == "ok":
                try:
                    messagebox.showinfo("导出结果", f"导出完成：{payload}", parent=host)
                except Exception:
                    pass
            else:
                try:
                    messagebox.showerror("导出结果", f"导出失败：{payload}", parent=host)
                except Exception:
                    pass

        try:
            host.after(100, _poll_result)
        except Exception:
            pass
        return None

    def _show_history_export_progress(self) -> tk.Toplevel:
        host = cast(tk.Tk, self)
        top = tk.Toplevel(host)
        top.title("导出结果")
        top.transient(host)
        try:
            top.grab_set()
        except Exception:
            pass
        try:
            top.resizable(False, False)
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
        try:
            top.update_idletasks()
            x = host.winfo_rootx() + max(0, (host.winfo_width() - top.winfo_width()) // 2)
            y = host.winfo_rooty() + max(0, (host.winfo_height() - top.winfo_height()) // 2)
            top.geometry(f"+{x}+{y}")
        except Exception:
            pass
        return top
