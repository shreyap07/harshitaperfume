# Perfume Business Analytics Dashboard

This project is designed for GitHub upload and Streamlit deployment.

## Files included
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - Project guide
- `DataanalysisPblIndividual_Ms25mm097.xlsx` - Source workbook used by the app

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How the app works
The app automatically reads the Excel workbook, detects the cleaned data sheet, cleans the usable rows, and builds:
- Classification model for high purchase intent
- Regression model for willingness to pay
- KMeans clustering for customer segmentation
- Association rule mining using support, confidence, and lift
- Interactive Plotly charts for all sections

## Streamlit deployment
1. Upload all files in this folder to the root of your GitHub repository.
2. Go to Streamlit Community Cloud.
3. Connect the GitHub repo.
4. Set the main file path as `app.py`.
5. Deploy.

## Notes
- The app also supports uploading an updated Excel file from the Streamlit sidebar.
- No subfolder structure is required for deployment.
