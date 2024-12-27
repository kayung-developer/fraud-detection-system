import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from customtkinter import CTk, CTkButton, CTkLabel, CTkEntry, CTkFrame, CTkTextbox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tf_keras.models import load_model

# Load pre-trained Autoencoder model
def load_autoencoder_model(model_path="autoencoder_model.h5"):
    return load_model(model_path)

# Detect anomalies in the uploaded data
def detect_anomalies(data, model, scaler, threshold=0.01):
    scaled_data = scaler.transform(data)
    reconstructed = model.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - reconstructed, 2), axis=1)
    anomalies = mse > threshold
    return anomalies, mse

# UI Application Class
class FraudDetectionApp(CTk):
    def __init__(self):
        super().__init__()
        self.title("AI-Powered Financial Fraud Detection")
        self.geometry("800x600")
        self.resizable(False, False)

        # Load pre-trained model and scaler
        self.autoencoder = load_autoencoder_model()
        self.scaler = StandardScaler()

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Header
        header = CTkLabel(self, text="Financial Fraud Detection System", font=("Arial", 20))
        header.pack(pady=10)

        # File Upload Section
        file_frame = CTkFrame(self, width=700, height=100)
        file_frame.pack(pady=10)
        file_label = CTkLabel(file_frame, text="Upload Transaction Data (CSV)", font=("Arial", 14))
        file_label.pack(pady=5)
        upload_button = CTkButton(file_frame, text="Upload CSV", hover_color="red",command=self.upload_file)
        upload_button.pack(pady=5)

        # Result Display Section
        result_frame = CTkFrame(self, width=700, height=300)
        result_frame.pack(pady=20)
        result_label = CTkLabel(result_frame, text="Results", font=("Arial", 16))
        result_label.pack(pady=5)

        self.result_textbox = CTkTextbox(result_frame, width=680, height=200, font=("Arial", 12))
        self.result_textbox.pack(pady=10)

        # Footer Buttons
        footer_frame = CTkFrame(self, width=700, height=50)
        footer_frame.pack(pady=20)
        analyze_button = CTkButton(footer_frame, text="Analyze Data", hover_color="red",command=self.analyze_data)
        analyze_button.pack(side=tk.LEFT, padx=20)
        quit_button = CTkButton(footer_frame, text="Exit", bg_color="green", hover_color="red", command=self.quit)
        quit_button.pack(side=tk.RIGHT, padx=20)

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")], title="Choose a CSV file"
        )
        if file_path:
            self.transaction_data = pd.read_csv(file_path)
            self.result_textbox.insert("1.0", f"File uploaded successfully: {file_path}\n")
        else:
            messagebox.showwarning("Warning", "No file selected!")

    def analyze_data(self):
        if not hasattr(self, "transaction_data"):
            messagebox.showerror("Error", "No transaction data uploaded!")
            return

        # Preprocess and detect anomalies
        try:
            numeric_data = self.transaction_data.select_dtypes(include=[np.number])
            self.scaler.fit(numeric_data)  # Fit scaler to the data
            anomalies, mse = detect_anomalies(numeric_data, self.autoencoder, self.scaler)

            # Add anomaly information to the dataframe
            self.transaction_data["Anomaly"] = anomalies
            self.transaction_data["MSE"] = mse

            # Display results
            anomaly_count = anomalies.sum()
            self.result_textbox.insert("1.0", f"\nDetected {anomaly_count} anomalies.\n")
            messagebox.showinfo("Analysis Complete", f"Analysis complete! {anomaly_count} anomalies detected.")

            # Save results to a new CSV
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Results"
            )
            if save_path:
                self.transaction_data.to_csv(save_path, index=False)
                self.result_textbox.insert("1.0", f"Results saved to {save_path}\n")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis:\n{e}")

# Run the application
if __name__ == "__main__":
    app = FraudDetectionApp()
    app.mainloop()
