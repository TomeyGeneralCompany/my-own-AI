"""Local desktop workbench: owner events only, one controller, no remote service."""
import json
from concurrent.futures import ThreadPoolExecutor
from .session import open_controller, _commands


class Workbench:
    def __init__(self, controller):
        self.controller = controller
        self.commands = _commands(controller)

    def call(self, command, **kwargs):
        if command not in self.commands:
            raise ValueError("Unknown owner command")
        result = self.commands[command](**kwargs)
        return {"result": result, "catalog": self.controller.mode_catalog()}


def launch(args):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title("Ancestor — operational workbench")
    root.geometry("900x680")
    root.minsize(650, 520)
    try:
        controller = open_controller(args)
    except BaseException:
        root.destroy()
        raise
    workbench = Workbench(controller)
    pool = ThreadPoolExecutor(max_workers=1)
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="ANCESTOR", font=("TkDefaultFont", 20, "bold")).pack(anchor="w")
    status = tk.StringVar(value="No mode active — select a module")
    ttk.Label(frame, textvariable=status).pack(anchor="w", pady=(0, 12))
    modes = ttk.Frame(frame)
    modes.pack(fill="x")
    mode_buttons = {}
    controls = []
    operation = tk.StringVar()
    combo = ttk.Combobox(frame, textvariable=operation, state="readonly")
    ttk.Label(frame, text="Operation").pack(anchor="w", pady=(12, 0))
    combo.pack(fill="x")
    ttk.Label(frame, text="Objective").pack(anchor="w")
    objective = ttk.Entry(frame)
    objective.pack(fill="x")
    ttk.Label(frame, text="Project name").pack(anchor="w")
    project = ttk.Entry(frame)
    project.insert(0, "default")
    project.pack(fill="x")
    ttk.Label(frame, text="Operation options (JSON)").pack(anchor="w")
    options = tk.Text(frame, height=4)
    options.insert("1.0", "{}")
    options.pack(fill="x")
    bar = ttk.Frame(frame)
    bar.pack(fill="x", pady=8)
    output = tk.Text(frame, wrap="word", state="disabled")
    output.pack(fill="both", expand=True)
    busy = False
    closing = False

    def show(value):
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("1.0", json.dumps(value, indent=2, ensure_ascii=False))
        output.configure(state="disabled")

    def refresh(catalog):
        active = catalog["active"]
        status.set("Active mode: " + next((m["label"] for m in catalog["modules"]
                   if m["id"] == active), "None") + " | " +
                   str(sum(m["loaded"] for m in catalog["modules"])) + " loaded")
        for m in catalog["modules"]:
            button = mode_buttons[m["id"]]
            button.configure(text=m["label"] + (" (active)" if m["loaded"] else ""),
                             state="disabled" if busy or m["loaded"] else "normal")
        ops = next((m["operations"] for m in catalog["modules"] if m["id"] == active), [])
        combo.configure(values=ops, state="disabled" if busy or not ops else "readonly")
        if operation.get() not in ops:
            operation.set(ops[0] if ops else "")
        for button in controls:
            button.configure(state="normal" if active and not busy else "disabled")

    def submit(command, **kwargs):
        nonlocal busy
        if busy or closing:
            return
        busy = True
        for button in [*mode_buttons.values(), *controls]:
            button.configure(state="disabled")
        combo.configure(state="disabled")
        status.set("Working: " + command)
        future = pool.submit(workbench.call, command, **kwargs)

        def poll():
            nonlocal busy
            if not future.done():
                root.after(60, poll)
                return
            busy = False
            try:
                reply = future.result()
                show(reply["result"])
                refresh(reply["catalog"])
            except Exception as exc:
                show({"error": str(exc)})
                refresh(controller.mode_catalog())
            if closing:
                finish()
        root.after(60, poll)

    def start():
        try:
            inputs = json.loads(options.get("1.0", "end"))
            if not isinstance(inputs, dict):
                raise ValueError("Options must be a JSON object")
            submit("start", operation=operation.get(), objective=objective.get(),
                   project=project.get(), inputs=inputs)
        except Exception as exc:
            show({"error":str(exc)})

    for i, m in enumerate(controller.mode_catalog()["modules"]):
        button = ttk.Button(modes, text=m["label"], command=lambda name=m["id"]: submit("switch", mode=name))
        button.grid(row=i//3, column=i%3, sticky="ew", padx=3, pady=3)
        modes.columnconfigure(i%3, weight=1)
        mode_buttons[m["id"]] = button
    for label, command in (("Start project", start), ("Advance one stage", lambda: submit("step")),
                           ("Resume / inspect", lambda: submit("progress")),
                           ("Save", lambda: submit("checkpoint"))):
        button = ttk.Button(bar, text=label, command=command)
        button.pack(side="left", padx=3)
        controls.append(button)

    def finish():
        try:
            pool.submit(controller.close).result()
        except Exception as exc:
            show({"error": str(exc)})
            status.set("Save failed; window remains open")
            return
        pool.shutdown(wait=True)
        root.destroy()

    def close():
        nonlocal closing
        closing = True
        if not busy:
            finish()

    root.protocol("WM_DELETE_WINDOW", close)
    refresh(controller.mode_catalog())
    root.mainloop()
