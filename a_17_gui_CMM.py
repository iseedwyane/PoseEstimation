import tkinter as tk
from tkinter import ttk

class CMMMeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CMM Measurement")
        
        # 放大窗口尺寸
        self.root.geometry("1024x768")

        # 全局样式设置，放大字体
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 15))
        style.configure("TButton", font=("Helvetica", 15))
        style.configure("TEntry", font=("Helvetica", 15))
        style.configure("TLabelFrame", font=("Helvetica", 15))

        # Machine Status Monitoring
        machine_frame = ttk.LabelFrame(root, text="Machine Status Monitoring")
        machine_frame.grid(row=0, column=0, padx=15, pady=15, sticky="ew")

        ttk.Button(machine_frame, text="StartSession").grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(machine_frame, text="Home").grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(machine_frame, text="ErrorClear").grid(row=0, column=2, padx=10, pady=10)
        ttk.Button(machine_frame, text="EndSession").grid(row=0, column=3, padx=10, pady=10)

        # Motion Parameters
        motion_frame = ttk.LabelFrame(root, text="Motion Parameters")
        motion_frame.grid(row=1, column=0, padx=15, pady=15, sticky="ew")

        labels = ["Goto Speed", "Goto Acceleration", "PtMeasure Speed", "PtMeasure Acceleration",
                  "PtMeasure Approach", "PtMeasure Search", "PtMeasure Retract"]
        default_values = ["100.0000", "100.0000", "20.0000", "30.0000", "50.0000", "100.0000", "100.0000"]
        self.motion_params = {}

        for i, (label, default) in enumerate(zip(labels, default_values)):
            ttk.Label(motion_frame, text=label).grid(row=i // 4, column=(i % 4) * 2, padx=10, pady=10)
            self.motion_params[label] = tk.StringVar(value=default)
            ttk.Entry(motion_frame, textvariable=self.motion_params[label], width=15).grid(row=i // 4, column=(i % 4) * 2 + 1, padx=10, pady=10)

        ttk.Button(motion_frame, text="SetMotionPara").grid(row=2, column=4, padx=10, pady=10)
        ttk.Button(motion_frame, text="GetMotionPara").grid(row=2, column=5, padx=10, pady=10)

        # GoTo Coordinates
        goto_frame = ttk.LabelFrame(root, text="GoTo Coordinates")
        goto_frame.grid(row=2, column=0, padx=15, pady=15, sticky="ew")

        coord_labels = ["X", "Y", "Z", "i", "j", "k"]
        self.coordinates = {}

        for i, label in enumerate(coord_labels):
            ttk.Label(goto_frame, text=label).grid(row=i // 3, column=(i % 3) * 2, padx=10, pady=10)
            self.coordinates[label] = tk.StringVar(value="0.0000")
            ttk.Entry(goto_frame, textvariable=self.coordinates[label], width=15).grid(row=i // 3, column=(i % 3) * 2 + 1, padx=10, pady=10)

        ttk.Button(goto_frame, text="GoTo").grid(row=0, column=6, padx=10, pady=10)
        ttk.Button(goto_frame, text="Stop All").grid(row=0, column=7, padx=10, pady=10)
        ttk.Button(goto_frame, text="Point Measure").grid(row=1, column=6, padx=10, pady=10)
        ttk.Button(goto_frame, text="GetCurrentXYZ").grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        ttk.Button(goto_frame, text="Multi Point Measure").grid(row=2, column=2, columnspan=2, padx=10, pady=10)
        ttk.Button(goto_frame, text="Registration").grid(row=2, column=4, columnspan=2, padx=10, pady=10)
        ttk.Button(goto_frame, text="Automatic Sampling Path").grid(row=2, column=6, columnspan=2, padx=10, pady=10)

        # Server Response Area
        response_frame = ttk.LabelFrame(root, text="CMM Server Response")
        response_frame.grid(row=3, column=0, padx=15, pady=15, sticky="ew")

        self.response_text = tk.Text(response_frame, height=10, width=90, font=("Helvetica", 15))
        self.response_text.pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = CMMMeasurementApp(root)
    root.mainloop()