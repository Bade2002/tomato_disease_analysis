import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomato Leaf Disease AI Classifier")
        self.root.geometry("850x700")
        self.root.minsize(800, 650)
        self.root.configure(bg="#0f172a")  # Modern dark theme

        # Hardware Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = r"C:\Users\prajw\Downloads\Potato-Leaf-Diseases-Detection-main\tomato_disease_analysis"
        self.labels_path = os.path.join(self.base_dir, "class_labels.json")

        # Load Class Labels
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r") as f:
                self.class_names = json.load(f)
        else:
            self.class_names = [f"Class_{i}" for i in range(10)]

        # Preprocessing Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.current_model = None
        self.current_image_path = None

        # Build Interface
        self.create_widgets()

        # Load default model
        self.on_model_change()

    def create_widgets(self):
        # Header Banner
        header_frame = tk.Frame(self.root, bg="#1e293b", pady=12)
        header_frame.pack(fill="x")

        title = tk.Label(
            header_frame,
            text="🍅 Tomato Leaf Disease Diagnostic Center",
            font=("Segoe UI", 16, "bold"),
            fg="#38bdf8",
            bg="#1e293b"
        )
        title.pack()

        device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        device_lbl = tk.Label(
            header_frame,
            text=f"Hardware Accelerator: {device_name} ({self.device.type.upper()})",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b"
        )
        device_lbl.pack(pady=(2, 0))

        # Main Content Layout (Two Columns)
        content_frame = tk.Frame(self.root, bg="#0f172a", padx=20, pady=15)
        content_frame.pack(fill="both", expand=True)

        # ----------------- Left Panel: Controls & Image -----------------
        left_panel = tk.Frame(content_frame, bg="#1e293b", bd=1, relief="flat", padx=15, pady=15)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Model Selector Dropdown
        model_select_lbl = tk.Label(left_panel, text="Select Neural Network Model:", font=("Segoe UI", 11, "bold"),
                                    fg="#f8fafc", bg="#1e293b")
        model_select_lbl.pack(anchor="w", pady=(0, 5))

        self.model_var = tk.StringVar(value="MobileNetV3 (Transfer Learning)")
        self.model_dropdown = ttk.Combobox(
            left_panel,
            textvariable=self.model_var,
            values=["MobileNetV3 (Transfer Learning)", "Custom CNN (From Scratch)"],
            state="readonly",
            font=("Segoe UI", 10)
        )
        self.model_dropdown.pack(fill="x", pady=(0, 15))
        self.model_dropdown.bind("<<ComboboxSelected>>", lambda e: self.on_model_change())

        # Image Preview Box
        self.img_label = tk.Label(
            left_panel,
            text="No Image Loaded\n\nClick 'Upload Leaf Image' below",
            font=("Segoe UI", 10),
            fg="#64748b",
            bg="#0f172a",
            width=35,
            height=14,
            relief="groove"
        )
        self.img_label.pack(fill="both", expand=True, pady=10)

        # Action Buttons Frame
        btn_frame = tk.Frame(left_panel, bg="#1e293b")
        btn_frame.pack(fill="x", pady=(10, 0))

        self.upload_btn = tk.Button(
            btn_frame,
            text="📂 Upload Leaf Image",
            command=self.upload_image,
            font=("Segoe UI", 11, "bold"),
            bg="#0284c7",
            fg="white",
            activebackground="#0369a1",
            activeforeground="white",
            relief="flat",
            pady=8,
            cursor="hand2"
        )
        self.upload_btn.pack(fill="x", pady=2)

        # ----------------- Right Panel: Prediction Results -----------------
        right_panel = tk.Frame(content_frame, bg="#1e293b", bd=1, relief="flat", padx=20, pady=15)
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        res_header = tk.Label(right_panel, text="Diagnostic Report", font=("Segoe UI", 13, "bold"), fg="#38bdf8",
                              bg="#1e293b")
        res_header.pack(anchor="w", pady=(0, 10))

        # Model Status Badge
        self.status_lbl = tk.Label(right_panel, text="Model: Ready", font=("Segoe UI", 9, "italic"), fg="#a855f7",
                                   bg="#1e293b")
        self.status_lbl.pack(anchor="w", pady=(0, 15))

        # Primary Diagnosis Frame
        diag_box = tk.Frame(right_panel, bg="#0f172a", padx=15, pady=12, relief="groove")
        diag_box.pack(fill="x", pady=(0, 15))

        self.diag_title = tk.Label(diag_box, text="Diagnosis: -", font=("Segoe UI", 12, "bold"), fg="#f8fafc",
                                   bg="#0f172a", wraplength=320, justify="left")
        self.diag_title.pack(anchor="w")

        self.diag_conf = tk.Label(diag_box, text="Confidence: -", font=("Segoe UI", 11, "bold"), fg="#22c55e",
                                  bg="#0f172a")
        self.diag_conf.pack(anchor="w", pady=(4, 0))

        # Top 3 Candidates Breakdown
        candidates_lbl = tk.Label(right_panel, text="Top Probability Candidates:", font=("Segoe UI", 11, "bold"),
                                  fg="#e2e8f0", bg="#1e293b")
        candidates_lbl.pack(anchor="w", pady=(10, 8))

        self.bars_frame = tk.Frame(right_panel, bg="#1e293b")
        self.bars_frame.pack(fill="both", expand=True)

        self.candidate_rows = []
        for i in range(3):
            row_frame = tk.Frame(self.bars_frame, bg="#1e293b", pady=6)
            row_frame.pack(fill="x")

            name_lbl = tk.Label(row_frame, text=f"{i + 1}. -", font=("Segoe UI", 9), fg="#cbd5e1", bg="#1e293b",
                                width=26, anchor="w")
            name_lbl.pack(side="left")

            prog = ttk.Progressbar(row_frame, orient="horizontal", length=120, mode="determinate")
            prog.pack(side="left", padx=8)

            pct_lbl = tk.Label(row_frame, text="0.0%", font=("Segoe UI", 9, "bold"), fg="#38bdf8", bg="#1e293b",
                               width=8, anchor="e")
            pct_lbl.pack(side="right")

            self.candidate_rows.append((name_lbl, prog, pct_lbl))

    # ---------------------------------------------------------
    # 3. Dynamic Model Loading
    # ---------------------------------------------------------
    def on_model_change(self):
        selection = self.model_var.get()
        self.status_lbl.config(text="Loading weights into GPU...", fg="#f59e0b")
        self.root.update_idletasks()

        if "MobileNetV3" in selection:
            weights_file = os.path.join(self.base_dir, "best_tomato_model.pth")
            if not os.path.exists(weights_file):
                messagebox.showerror("File Error", f"Cannot find weights at:\n{weights_file}")
                return

            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(self.class_names))
            checkpoint = torch.load(weights_file, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.current_model = model.to(self.device).eval()
            self.status_lbl.config(text=f"Active: MobileNetV3 | Val Acc: {checkpoint.get('accuracy', 0) * 100:.2f}%",
                                   fg="#22c55e")

        else:  # Custom CNN
            weights_file = os.path.join(self.base_dir, "custom_tomato_cnn_best.pth")
            if not os.path.exists(weights_file):
                messagebox.showwarning("File Missing",
                                       f"Custom CNN weights not found ({weights_file}). Please run train_custom_cnn.py first.")
                self.status_lbl.config(text="Custom weights not found!", fg="#ef4444")
                return

            model = CustomTomatoCNN(num_classes=len(self.class_names))
            checkpoint = torch.load(weights_file, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.current_model = model.to(self.device).eval()
            self.status_lbl.config(text=f"Active: Custom CNN | Val Acc: {checkpoint.get('accuracy', 0) * 100:.2f}%",
                                   fg="#22c55e")

        # If an image is already uploaded, immediately re-run inference with the newly selected model
        if self.current_image_path:
            self.run_inference()

    # ---------------------------------------------------------
    # 4. Image Upload & Inference
    # ---------------------------------------------------------
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Tomato Leaf Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.JPG;*.PNG")]
        )
        if not file_path:
            return

        self.current_image_path = file_path

        # Render preview image
        img = Image.open(file_path).convert("RGB")
        preview = img.copy()
        preview.thumbnail((320, 260))
        img_tk = ImageTk.PhotoImage(preview)

        self.img_label.config(image=img_tk, text="")
        self.img_label.image = img_tk

        self.run_inference()

    def run_inference(self):
        if not self.current_image_path or self.current_model is None:
            return

        img = Image.open(self.current_image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.current_model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            topk_probs, topk_indices = torch.topk(probabilities, k=3)

        top_class = self.class_names[topk_indices[0]].replace('_', ' ')
        top_prob = topk_probs[0].item() * 100

        # Update Primary Output
        self.diag_title.config(text=f"Diagnosis: {top_class}")
        self.diag_conf.config(text=f"Confidence: {top_prob:.2f}%")

        # Update Top 3 Rows & Meters
        for i in range(3):
            name_lbl, prog, pct_lbl = self.candidate_rows[i]
            c_name = self.class_names[topk_indices[i]].replace('_', ' ')
            c_prob = topk_probs[i].item() * 100

            name_lbl.config(text=f"{i + 1}. {c_name[:22]}")
            prog['value'] = c_prob
            pct_lbl.config(text=f"{c_prob:.1f}%")


# =====================================================================
# 3. Application Launcher (Persistent Event Loop)
# =====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    # mainloop keeps the application running persistently until the X button is clicked
    root.mainloop()
