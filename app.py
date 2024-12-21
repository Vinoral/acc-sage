from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
from sqlalchemy import func
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
from datetime import datetime  # Add this import at the top
import matplotlib
matplotlib.use('Agg')
plt.style.use('fivethirtyeight')


# Initialize Flask and configure database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///financial_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize SQLAlchemy and Migrate
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load the trained model
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Feature columns expected by the model
feature_columns = [
    "ROA(C) before interest and depreciation before interest",
    "ROA(A) before interest and % after tax",
    "ROA(B) before interest and depreciation after tax",
    "Operating Gross Margin",
    "Tax rate (A)",
    "Net Value Per Share (B)",
    "Net Value Per Share (A)",
    "Net Value Per Share (C)",
    "Persistent EPS in the Last Four Seasons",
    "Current Ratio",
    "Quick Ratio",
    "Debt ratio %",
    "Net worth/Assets",
    "Operating profit/Paid-in capital",
    "Net profit before tax/Paid-in capital",
    "Working Capital to Total Assets",
    "Quick Assets/Current Liability",
    "Current Liability to Assets",
    "Total income/Total expense",
    "Current Liability to Current Assets",
    "Net Income to Total Assets",
    "Equity to Liability",
    "Cash Ratio"
]


# Database Model
class FinancialData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.Integer, nullable=True)
    
    # Dynamically create columns for all features
    for col in feature_columns:
        # Replace spaces and special characters to create valid column names
        safe_col = col.replace('(', '_').replace(')', '').replace(' ', '_').replace('%', 'Percent')
        locals()[safe_col] = db.Column(db.Float, nullable=True)

    def to_dict(self):
        # Convert model instance to dictionary
        return {
            'id': self.id,
            'prediction': self.prediction,
            **{
                col.replace('(', '_').replace(')', '').replace(' ', '_').replace('%', 'Percent'): 
                getattr(self, col.replace('(', '_').replace(')', '').replace(' ', '_').replace('%', 'Percent')) 
                for col in feature_columns
            }
        }

    @classmethod
    def create_from_row(cls, row, prediction):
        # Dynamic column name conversion
        def safe_column_name(col):
            return col.replace('(', '_').replace(')', '').replace(' ', '_').replace('%', 'Percent')

        # Prepare data for database insertion
        record_data = {
            'prediction': prediction,
            **{safe_column_name(col): row.get(col) for col in feature_columns}
        }

        return cls(**record_data)

@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    if request.method == "POST":
        try:
            uploaded_file = request.files.get("file")
            if not uploaded_file:
                return "Tidak ada file yang diunggah!"

            # Read CSV
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            
            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Prepare scaler
            scaler = MinMaxScaler()
            
            # Ensure all required columns exist
            for col in feature_columns:
                safe_col = col.lower()
                if safe_col not in df.columns:
                    df[safe_col] = 0
            
            # Select and scale data
            data = df[[col.lower() for col in feature_columns]].astype(float)
            data_scaled = scaler.fit_transform(data)
            
            # Predict
            predictions = model.predict(data_scaled)
            
            # Save to database with predictions
            financial_records = []
            for idx, (_, row) in enumerate(data.iterrows()):
                # Convert row to dictionary with original column names
                row_dict = {feature_columns[i]: row.values[i] for i in range(len(feature_columns))}
                
                # Create and save record
                record = FinancialData.create_from_row(row_dict, int(predictions[idx]))
                financial_records.append(record)
            
            # Bulk insert
            db.session.add_all(financial_records)
            db.session.commit()
            
            return redirect(url_for('model_info'))
        
        except Exception as e:
            return f"Terjadi error: {str(e)}"
    
    return render_template("prediksi.html")

@app.route("/try-new-data", methods=["GET", "POST"])
def try_new_data():
    prediction = None
    if request.method == "POST":
        try:
            # Collect input data
            input_data = {feature: float(request.form.get(feature, 0)) for feature in feature_columns}

            # Convert to DataFrame
            df = pd.DataFrame([input_data])

            # Scale the data
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(df)

            # Predict
            prediction = int(model.predict(data_scaled)[0])

            # Save to database
            record_data = {
                col.replace('(', '_').replace(')', '').replace(' ', '_').replace('%', 'Percent'): 
                value for col, value in input_data.items()
            }
            record_data['prediction'] = prediction
            
            # Create and save record
            new_record = FinancialData(**record_data)
            db.session.add(new_record)
            db.session.commit()

        except Exception as e:
            return f"Terjadi error: {str(e)}"

    return render_template("try_new_data.html", feature_columns=feature_columns, prediction=prediction)




@app.route("/visualization")
def visualization():
    try:
        # Fetch all records from database
        records = FinancialData.query.all()
        
        if not records:
            return render_template("visualization.html", error="No data available for visualization")

        # Count predictions
        prediction_counts = {}
        for record in records:
            pred = record.prediction
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1

        # Prepare data for plotting
        labels = list(prediction_counts.keys())
        values = list(prediction_counts.values())

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Pie Chart
        colors = ['#ff9999', '#66b3ff']
        ax1.pie(values, 
                labels=[f'Class {label}' for label in labels],
                colors=colors,
                autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Distribution of Predictions')

        # Bar Chart
        bars = ax2.bar(labels, values, color=colors)
        ax2.set_title('Number of Predictions by Class')
        ax2.set_xlabel('Prediction Class')
        ax2.set_ylabel('Count')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        # Adjust layout
        plt.tight_layout()

        # Ensure the directory exists
        os.makedirs("static/images", exist_ok=True)
        
        # Save the plot
        vis_path = "static/images/prediction_visualization.png"
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()

        return render_template("visualization.html", 
                            vis_path=vis_path,
                            timestamp=datetime.now().timestamp())

    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        return render_template("visualization.html", 
                            error=f"Error generating visualization: {str(e)}")


@app.route("/model")
def model_info():
    # Fetch records with predictions
    records = FinancialData.query.all()
    
    # Total records and prediction distribution
    total_records = len(records)
    
    # Count predictions
    prediction_counts = {}
    for record in records:
        pred = record.prediction
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    return render_template(
        "model.html", 
        records=records, 
        total_records=total_records,
        prediction_counts=prediction_counts
    )

if __name__ == "__main__":
    # Ensure static/images folder exists
    os.makedirs("static/images", exist_ok=True)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)