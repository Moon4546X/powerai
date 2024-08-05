import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class PowerAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Power AI")
        self.create_widgets()

    def create_widgets(self):
        # Create frames
        self.frame1 = tk.Frame(self.root, width=640)
        self.frame2 = tk.Frame(self.root, width=160)
        
        self.frame1.grid(row=0, column=0, sticky='ns')
        self.frame2.grid(row=0, column=1, sticky='ns')
        
        # Top bar with about button
        self.top_bar = tk.Frame(self.root, bg='#f0f0f0', height=30)
        self.top_bar.grid(row=1, column=0, columnspan=2, sticky='ew')
        
        self.title_label = tk.Label(self.top_bar, text="Power AI", bg='#f0f0f0', fg='#000', font=('Arial', 16, 'bold'))
        self.title_label.pack(side='left', padx=10)
        
        self.about_button = tk.Button(self.top_bar, text="About", command=self.redirect_about)
        self.about_button.pack(side='right', padx=10)
        
        # File upload button
        self.upload_button = tk.Button(self.frame1, text="Upload CSV/Excel", command=self.upload_file)
        self.upload_button.pack(pady=10)
        
        # Data display
        self.data_display = tk.Text(self.frame1, height=20)
        self.data_display.pack(fill='both', expand=True)
        
        # Filters
        self.filter_frame = tk.Frame(self.frame1)
        self.filter_frame.pack(pady=10, fill='x')
        
        tk.Label(self.filter_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, padx=5, pady=5)
        self.start_date_entry = tk.Entry(self.filter_frame)
        self.start_date_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.filter_frame, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5)
        self.end_date_entry = tk.Entry(self.filter_frame)
        self.end_date_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Button(self.filter_frame, text="Apply Filters", command=self.apply_filters).grid(row=2, columnspan=2, pady=10)
        
        # Button to open interactive graph
        self.graph_button = tk.Button(self.frame2, text="Open Interactive Graph", command=self.open_graph)
        self.graph_button.pack(pady=10)
        
        # Seasonality display
        self.seasonality_display = tk.Text(self.frame2, height=10)
        self.seasonality_display.pack(fill='both', expand=True)
        
        # Save data and graph
        self.save_button = tk.Button(self.frame2, text="Save Data", command=self.save_data)
        self.save_button.pack(pady=10)
        
        self.save_graph_button = tk.Button(self.frame2, text="Save Graph", command=self.save_graph)
        self.save_graph_button.pack(pady=10)
        
        # Machine learning prediction
        self.ml_button = tk.Button(self.frame2, text="Predict Best Selling Product", command=self.predict_best_selling_product)
        self.ml_button.pack(pady=10)
        
        self.ml_display = tk.Text(self.frame2, height=10)
        self.ml_display.pack(fill='both', expand=True)

    def redirect_about(self):
        webbrowser.open("https://achintya-iota.vercel.app")

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.df = pd.read_excel(file_path)
                self.df['Date'] = pd.to_datetime(self.df['Date'])  # Convert 'Date' column to datetime
                self.filtered_df = self.df.copy()  # Make a copy of the original dataframe
                self.display_data()
                self.detect_seasonality()
                self.generate_graph()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {str(e)}")
        else:
            messagebox.showwarning("No file selected", "Please select a CSV or Excel file to upload.")

    def display_data(self):
        try:
            product_counts = self.filtered_df['Product'].value_counts()
            self.data_display.delete(1.0, tk.END)
            self.data_display.insert(tk.END, product_counts.to_string())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display data: {str(e)}")

    def apply_filters(self):
        try:
            self.filtered_df = self.df.copy()  # Reset to the original dataframe before applying filters
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            if start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                self.filtered_df = self.filtered_df[self.filtered_df['Date'] >= start_date]
            if end_date:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
                self.filtered_df = self.filtered_df[self.filtered_df['Date'] <= end_date]
            self.display_data()
            self.generate_graph()
            self.detect_seasonality()
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD.")

    def generate_graph(self):
        try:
            self.filtered_df['Date'] = pd.to_datetime(self.filtered_df['Date'])
            self.filtered_df['Month'] = self.filtered_df['Date'].dt.month
            monthly_data = self.filtered_df.groupby(['Month', 'Product'])['Units Sold'].sum().unstack().fillna(0)
            
            fig = make_subplots(rows=1, cols=1)
            for product in monthly_data.columns:
                fig.add_trace(go.Bar(x=monthly_data.index, y=monthly_data[product], name=product))

            fig.update_layout(barmode='stack', title='Monthly Units Sold by Product', xaxis_title='Month', yaxis_title='Units Sold')

            # Save the plot as an HTML file
            self.plot_file = "plot.html"
            fig.write_html(self.plot_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate graph: {str(e)}")

    def open_graph(self):
        if hasattr(self, 'plot_file'):
            webbrowser.open(self.plot_file)
        else:
            messagebox.showwarning("No Graph", "Please upload a file first to generate a graph.")

    def save_data(self):
        if hasattr(self, 'filtered_df'):
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.filtered_df.to_csv(file_path, index=False)
        else:
            messagebox.showwarning("No Data", "Please upload a file first to save data.")

    def save_graph(self):
        if hasattr(self, 'plot_file'):
            file_path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html")])
            if file_path:
                with open(self.plot_file, "r") as src_file:
                    with open(file_path, "w") as dest_file:
                        dest_file.write(src_file.read())
        else:
            messagebox.showwarning("No Graph", "Please generate a graph first to save it.")

    def detect_seasonality(self):
        try:
            self.filtered_df['Date'] = pd.to_datetime(self.filtered_df['Date'])
            self.filtered_df['Month'] = self.filtered_df['Date'].dt.month
            seasonal_data = self.filtered_df.groupby('Month')['Units Sold'].sum()
            
            self.seasonality_display.delete(1.0, tk.END)
            self.seasonality_display.insert(tk.END, seasonal_data.to_string())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect seasonality: {str(e)}")

    def predict_best_selling_product(self):
        try:
            df = self.filtered_df.copy()
            df['Month'] = df['Date'].dt.month
            X = df[['Month']]
            y = df['Product']
            
            # Encoding the labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            
            # Train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Determine the best-selling product for each month
            df['Prediction'] = le.inverse_transform(model.predict(df[['Month']]))
            best_selling_products = df.groupby('Month')['Prediction'].agg(lambda x: x.value_counts().index[0])
            
            # Display the results
            self.ml_display.delete(1.0, tk.END)
            self.ml_display.insert(tk.END, f"Accuracy: {accuracy:.2f}\n")
            self.ml_display.insert(tk.END, best_selling_products.to_string())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict best selling product: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PowerAIApp(root)
    root.mainloop()
