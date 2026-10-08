import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Mapping dictionaries based on dataset characteristics
mappings = {
    'buying': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
    'maint': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
    'doors': {'2': 2, '3': 3, '4': 4, '5more': 5},
    'persons': {'2': 2, '4': 4, 'more': 5},
    'lug_boot': {'small': 1, 'med': 2, 'big': 3},
    'safety': {'low': 1, 'med': 2, 'high': 3}
}

# User-friendly names for prediction classes
class_mapping_rev = {
    'unacc': 'Unacceptable ❌',
    'acc': 'Acceptable ✔️',
    'good': 'Good ⭐',
    'vgood': 'Very Good 🌟'
}

class CarEvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Safety & Acceptability Predictor")
        self.root.geometry("450x450")
        self.root.configure(padx=20, pady=20)
        
        self.model = None
        
        # First layout widgets
        self.create_widgets()
        
        # Delay the model training slightly so the GUI can show up first
        self.root.after(100, self.train_model)

    def train_model(self):
        try:
            self.result_label.config(text="Status: Downloading data & training model...", fg="blue")
            self.root.update()
            
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
            col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
            df = pd.read_csv(url, names=col_names)
            
            X = df.drop('class', axis=1)
            y = df['class']
            
            for col in mappings:
                X[col] = X[col].map(mappings[col])
                
            self.model = DecisionTreeClassifier(criterion='gini', random_state=42)
            self.model.fit(X, y)
            
            self.result_label.config(text="Status: Model Ready! Waiting for input.", fg="green")
            self.predict_btn.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.result_label.config(text="Status: Error loading model.", fg="red")
            
    def create_widgets(self):
        title_label = tk.Label(self.root, text="🚗 Car Evaluation Predictor", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 15))
        
        self.inputs = {}
        
        # Form frame
        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        
        row = 0
        for feature, mapping in mappings.items():
            lbl = tk.Label(frame, text=feature.replace('_', ' ').title() + ":", font=("Helvetica", 11))
            lbl.grid(row=row, column=0, padx=10, pady=8, sticky='e')
            
            combo = ttk.Combobox(frame, values=list(mapping.keys()), state="readonly", width=15)
            combo.set(list(mapping.keys())[0])
            combo.grid(row=row, column=1, padx=10, pady=8)
            self.inputs[feature] = combo
            
            row += 1
            
        self.predict_btn = tk.Button(self.root, text="Predict", font=("Helvetica", 12, "bold"), 
                                     bg="#0078D7", fg="white", state="disabled", command=self.predict, width=15)
        self.predict_btn.pack(pady=20)
        
        self.result_label = tk.Label(self.root, text="Status: Initializing...", font=("Helvetica", 12))
        self.result_label.pack(pady=5)
        
    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Model is not trained.")
            return
            
        input_data = []
        for feature in mappings.keys():
            val = self.inputs[feature].get()
            mapped_val = mappings[feature][val]
            input_data.append(mapped_val)
            
        # Perform prediction
        prediction = self.model.predict([input_data])[0]
        friendly_result = class_mapping_rev.get(prediction, prediction)
        
        self.result_label.config(text=f"Prediction: {friendly_result}", fg="#333333", font=("Helvetica", 14, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = CarEvaluationApp(root)
    root.mainloop()
