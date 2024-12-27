import pandas as pd
import numpy as np
from faker import Faker
import customtkinter as ctk
from tkinter import messagebox


# Function to generate synthetic data
def generate_synthetic_data(num_records=1000):
    faker = Faker()
    data = {
        "transaction_id": [faker.uuid4() for _ in range(num_records)],
        "account_id": [faker.uuid4() for _ in range(num_records)],
        "transaction_time": [faker.date_time_this_year() for _ in range(num_records)],
        "amount": np.random.exponential(scale=50, size=num_records).round(2),
        "location": [faker.city() for _ in range(num_records)],
        "category": np.random.choice(["Food", "Travel", "Shopping", "Healthcare"], num_records),
    }
    df = pd.DataFrame(data)

    # Inject anomalies
    anomalies = np.random.choice(num_records, size=int(0.05 * num_records), replace=False)
    df.loc[anomalies, "amount"] *= np.random.randint(10, 50, size=len(anomalies))
    return df


# GUI application class
class SyntheticDataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic Data Generator")
        self.root.geometry("400x250")

        # Title Label
        self.title_label = ctk.CTkLabel(root, text="Synthetic Data Generator", font=("Arial", 16))
        self.title_label.pack(pady=10)

        # Number of Records Label and Entry
        self.records_label = ctk.CTkLabel(root, text="Number of Records:")
        self.records_label.pack(pady=5)
        self.records_entry = ctk.CTkEntry(root, width=200)
        self.records_entry.pack(pady=5)

        # Generate Button
        self.generate_button = ctk.CTkButton(root, text="Generate Data", command=self.generate_data)
        self.generate_button.pack(pady=10)

        # Save Button
        self.save_button = ctk.CTkButton(root, text="Save to CSV", command=self.save_to_csv, state="disabled")
        self.save_button.pack(pady=10)

        self.data = None

    def generate_data(self):
        try:
            num_records = int(self.records_entry.get())
            if num_records <= 0:
                raise ValueError("Number of records must be positive.")

            self.data = generate_synthetic_data(num_records)
            messagebox.showinfo("Success", f"Generated {num_records} records successfully!")
            self.save_button.configure(state="normal")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def save_to_csv(self):
        if self.data is not None:
            self.data.to_csv("synthetic_transactions.csv", index=False)
            messagebox.showinfo("Success", "Data saved to 'synthetic_transactions.csv' successfully!")


# Initialize CustomTkinter theme
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

# Create the main window
root = ctk.CTk()
app = SyntheticDataApp(root)
root.mainloop()
