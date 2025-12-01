import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests

API_URL = "http://127.0.0.1:8000/predict_url"

def analyze_url():
    url = url_entry.get().strip()
    if not url:
        messagebox.showerror("Input Error", "Please enter a valid URL.")
        return

    try:
        analyze_button.config(state="disabled")
        root.update_idletasks()

        response = requests.post(API_URL, json={"url": url})
        analyze_button.config(state="normal")

        if response.status_code != 200:
            messagebox.showerror("Server Error", response.text)
            return

        data = response.json()

        label_value.config(text=data["label"])
        confidence_value.config(text=f"{data['confidence']:.3f}")

        if data["label"] == "Credible":
            label_value.config(foreground="#0a7c21")
        else:
            label_value.config(foreground="#b30000")

        preview_box.delete(1.0, tk.END)
        preview_box.insert(tk.END, data["preview"])

    except Exception as e:
        analyze_button.config(state="normal")
        messagebox.showerror("Connection Error", f"Failed to connect to API:\n\n{e}")


root = tk.Tk()
root.title("Medical Misinformation Detection System")
root.geometry("760x620")
root.configure(bg="#f4f6f8")

root.option_add("*Font", "Arial 11")

title_label = tk.Label(
    root,
    text="Medical Misinformation Detection",
    font=("Arial", 20, "bold"),
    bg="#f4f6f8",
    fg="#1f3b58"
)
title_label.pack(pady=(20, 10))

subtitle_label = tk.Label(
    root,
    text="Automated credibility evaluation for online health information",
    font=("Arial", 12),
    bg="#f4f6f8",
    fg="#4a4a4a"
)
subtitle_label.pack(pady=(0, 20))

input_frame = tk.Frame(root, bg="#f4f6f8")
input_frame.pack(pady=10)

url_label = tk.Label(input_frame, text="Website URL:", bg="#f4f6f8", fg="#1f3b58", font=("Arial", 12, "bold"))
url_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

url_entry = tk.Entry(input_frame, width=60, font=("Arial", 11))
url_entry.grid(row=0, column=1, padx=10, pady=5)

analyze_button = ttk.Button(input_frame, text="Analyze", command=analyze_url, width=15)
analyze_button.grid(row=0, column=2, padx=5, pady=5)

results_card = tk.Frame(root, bg="white", bd=1, relief="solid")
results_card.pack(fill="x", padx=25, pady=20)

header = tk.Label(
    results_card,
    text="Credibility Assessment",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="#1f3b58"
)
header.pack(anchor="w", padx=20, pady=(15, 5))

separator = ttk.Separator(results_card, orient='horizontal')
separator.pack(fill='x', padx=20, pady=5)

info_frame = tk.Frame(results_card, bg="white")
info_frame.pack(padx=20, pady=10, anchor="w")

tk.Label(info_frame, text="Assessment:", font=("Arial", 12, "bold"), bg="white", fg="#333").grid(row=0, column=0, sticky="w", pady=4)
label_value = tk.Label(info_frame, text="—", font=("Arial", 12), bg="white")
label_value.grid(row=0, column=1, sticky="w", padx=10, pady=4)

tk.Label(info_frame, text="Confidence:", font=("Arial", 12, "bold"), bg="white", fg="#333").grid(row=1, column=0, sticky="w", pady=4)
confidence_value = tk.Label(info_frame, text="—", font=("Arial", 12), bg="white")
confidence_value.grid(row=1, column=1, sticky="w", padx=10, pady=4)

preview_label = tk.Label(
    root,
    text="Extracted Text Preview",
    font=("Arial", 14, "bold"),
    bg="#f4f6f8",
    fg="#1f3b58"
)
preview_label.pack(anchor="w", padx=30)

preview_box = scrolledtext.ScrolledText(root, width=90, height=12, font=("Arial", 10))
preview_box.pack(padx=25, pady=10)

root.mainloop()



