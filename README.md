# AI-Powered Financial Fraud Detection System

This project implements an AI-powered financial fraud detection system using unsupervised machine learning techniques. The system leverages an autoencoder model to detect anomalies in transaction data, which may indicate fraudulent activities.

## Features

- **Anomaly Detection**: Detects fraudulent transactions by analyzing anomalies in transaction data using an autoencoder neural network.
- **Pre-trained Autoencoder Model**: An autoencoder model is trained on transaction data and used to predict anomalies.
- **User-Friendly UI**: Built with CustomTkinter to provide a clean and interactive interface for uploading transaction data and viewing results.
- **CSV File Support**: Upload and save transaction data in CSV format.
- **Scalable**: Uses StandardScaler to scale the data before prediction.
- **Save Results**: After detecting anomalies, results are saved to a new CSV file with anomaly flags.

---

## Getting Started

### Prerequisites

- Python 3.6+
- Dependencies: `tensorflow`, `customtkinter`, `pandas`, `numpy`, `scikit-learn`

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/kayung-developer/fraud-detection-system.git
    cd fraud-detection-system
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download or prepare a financial transaction dataset. The dataset should be in CSV format, containing numerical transaction features.
4. Generate synthetic data to test the fraud detection system
4. Train the Autoencoder model or use a pre-trained model (`autoencoder_model.h5`). See the **Training the Autoencoder** section below.

### Usage

1. Run the Python script to launch the application:

    ```bash
    python main.py
    ```

2. **Upload Data**: Click on the "Upload CSV" button to select a transaction dataset.
3. **Analyze Data**: Click on the "Analyze Data" button to detect anomalies.
4. **Save Results**: After the analysis, you can save the results to a new CSV file with anomaly flags.

---

## Generating Transaction Synthetic Data
You can generate synthetic transaction data for testing the fraud detection system using the following script. This will generate a random dataset and save it as `synthetic_transaction.csv`.


```bash
   python synthetic_data.py
 ```
```python
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
```







## Training the Autoencoder

To train the autoencoder model on your dataset, use the following script. This will save the trained model as `autoencoder_model.h5`.

```bash
   python train_autoencoder.py
 ```

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.optimizers import Adam
import customtkinter as ctk
from tkinter import filedialog, messagebox

class AutoencoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Autoencoder Trainer")
        self.root.geometry("500x400")

        # Title Label
        self.title_label = ctk.CTkLabel(root, text="Autoencoder Training Application", font=("Arial", 16))
        self.title_label.pack(pady=10)

        # File Selection Button
        self.file_button = ctk.CTkButton(root, text="Select CSV File", command=self.load_file)
        self.file_button.pack(pady=10)

        # Epochs Input
        self.epochs_label = ctk.CTkLabel(root, text="Number of Epochs:")
        self.epochs_label.pack(pady=5)
        self.epochs_entry = ctk.CTkEntry(root, width=200)
        self.epochs_entry.pack(pady=5)

        # Train Button
        self.train_button = ctk.CTkButton(root, text="Train Autoencoder", command=self.train_autoencoder, state="disabled")
        self.train_button.pack(pady=10)

        self.data = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", f"Loaded data from {file_path}")
                self.train_button.configure(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def train_autoencoder(self):
        try:
            if self.data is None:
                raise ValueError("No data loaded.")

            epochs = int(self.epochs_entry.get())
            if epochs <= 0:
                raise ValueError("Number of epochs must be positive.")

            numeric_data = self.data.select_dtypes(include=[np.number])
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)

            X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)
            input_dim = X_train.shape[1]

            autoencoder = Sequential([
                Dense(64, activation="relu", input_dim=input_dim),
                Dense(32, activation="relu"),
                Dense(64, activation="relu"),
                Dense(input_dim, activation="linear")
            ])

            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

            autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=1
            )

            autoencoder.save("autoencoder_model.h5")
            messagebox.showinfo("Success", "Autoencoder model saved as autoencoder_model.h5")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize CustomTkinter theme
ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

# Create the main window
root = ctk.CTk()
app = AutoencoderApp(root)
root.mainloop()


```

## Notes
Make sure to replace transaction_data.csv with your own dataset if needed.
The autoencoder model file (autoencoder_model.h5) should be located in the same directory as the UI script for proper loading.
You can customize the UI or training process according to your specific requirements.