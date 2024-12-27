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
