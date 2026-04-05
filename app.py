from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    import streamlit as st
except ImportError:  # lightweight fallback for local smoke tests
    class _DummySidebar:
        def file_uploader(self, *args, **kwargs):
            return None
        def success(self, *args, **kwargs):
            return None
        def info(self, *args, **kwargs):
            return None
        def write(self, *args, **kwargs):
            return None

    class _DummyTab:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    class _DummyColumn(_DummyTab):
        def metric(self, *args, **kwargs):
            return None

    class _DummyStreamlit:
        sidebar = _DummySidebar()
        def set_page_config(self, *args, **kwargs):
            return None
        def cache_data(self, show_spinner=False):
            def decorator(func):
                return func
            return decorator
        def title(self, *args, **kwargs):
            return None
        def caption(self, *args, **kwargs):
            return None
        def tabs(self, names):
            return [_DummyTab() for _ in names]
        def columns(self, n):
            return [_DummyColumn() for _ in range(n)]
        def markdown(self, *args, **kwargs):
            return None
        def subheader(self, *args, **kwargs):
            return None
        def write(self, *args, **kwargs):
            return None
        def plotly_chart(self, *args, **kwargs):
            return None
        def dataframe(self, *args, **kwargs):
            return None
        def metric(self, *args, **kwargs):
            return None
        def info(self, *args, **kwargs):
            return None
        def warning(self, *args, **kwargs):
            return None
        def error(self, *args, **kwargs):
            raise RuntimeError(args[0] if args else 'Streamlit error')
        def stop(self):
            raise SystemExit
        def slider(self, label, min_value=None, max_value=None, value=None, step=None):
            return value
        def download_button(self, *args, **kwargs):
            return None

    st = _DummyStreamlit()
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:  # allows local smoke tests before package install
    MLXTEND_AVAILABLE = False
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title='Perfume Business Analytics Dashboard', page_icon='💄', layout='wide')

EXPECTED_COLS = {
    'Customer Id',
    'Age',
    'Gender',
    'City',
    'Monthly Income',
    'Preferred Fragrance',
    'Shopping Channel',
    'Purchase Frequency per year',
    'Willingness to pay',
    'Influencer Impact',
    'Brand Awareness',
    'Purchase likelihood',
}

NUMERIC_COLS = [
    'Age',
    'Monthly Income',
    'Purchase Frequency per year',
    'Willingness to pay',
    'Influencer Impact',
    'Brand Awareness',
    'Purchase likelihood',
]

CATEGORICAL_COLS = [
    'Gender',
    'City',
    'Preferred Fragrance',
    'Shopping Channel',
    'Age group',
    'Income segment',
]


def find_default_excel() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    xlsx_files = [p for p in here.glob('*.xlsx') if not p.name.startswith('~$')]
    if not xlsx_files:
        return None
    preferred = [p for p in xlsx_files if 'DataanalysisPblIndividual' in p.name]
    return preferred[0] if preferred else xlsx_files[0]


@st.cache_data(show_spinner=False)
def load_business_data(uploaded_bytes: Optional[bytes] = None, filename: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if uploaded_bytes is not None and filename:
        temp_path = Path('/tmp') / filename
        temp_path.write_bytes(uploaded_bytes)
        source = temp_path
    else:
        source = find_default_excel()
        if source is None:
            raise FileNotFoundError('No Excel file found in the app folder.')

    xls = pd.ExcelFile(source)
    candidate_sheets = []
    for sheet in xls.sheet_names:
        for header_row in (0, 1, 2, 3):
            try:
                test_df = pd.read_excel(source, sheet_name=sheet, header=header_row)
                test_df.columns = [str(c).strip() for c in test_df.columns]
                overlap = len(EXPECTED_COLS.intersection(set(test_df.columns)))
                if overlap >= 8:
                    clean_bonus = 5 if 'clean' in sheet.lower() else 0
                    candidate_sheets.append((overlap + clean_bonus, overlap, sheet, header_row))
            except Exception:
                continue

    if not candidate_sheets:
        raise ValueError('Could not find a usable data sheet in the workbook.')

    _, _, sheet_name, header_row = sorted(candidate_sheets, reverse=True)[0]
    df = pd.read_excel(source, sheet_name=sheet_name, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    if 'Customer Id' in df.columns:
        df = df[pd.to_numeric(df['Customer Id'], errors='coerce').notna()].copy()
        df['Customer Id'] = pd.to_numeric(df['Customer Id'], errors='coerce').astype(int)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean text columns
    for col in [c for c in df.columns if c not in NUMERIC_COLS and c != 'Customer Id']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        if col in ['Gender', 'City', 'Preferred Fragrance', 'Shopping Channel', 'Age group', 'Income segment']:
            df[col] = df[col].str.title()

    # Drop rows missing core fields
    core_needed = ['Age', 'Gender', 'City', 'Monthly Income', 'Preferred Fragrance', 'Shopping Channel',
                   'Purchase Frequency per year', 'Willingness to pay', 'Influencer Impact', 'Brand Awareness', 'Purchase likelihood']
    existing_core = [c for c in core_needed if c in df.columns]
    df = df.dropna(subset=existing_core).copy()

    # Ensure derived segmentation columns exist
    if 'Age group' not in df.columns:
        df['Age group'] = pd.cut(
            df['Age'], bins=[0, 24, 30, 36, 100], labels=['18-24', '25-30', '31-36', '37-45'], include_lowest=True
        ).astype(str)
    if 'Income segment' not in df.columns:
        df['Income segment'] = pd.cut(
            df['Monthly Income'], bins=[0, 49999, 89999, float('inf')], labels=['Low', 'Middle', 'High'], include_lowest=True
        ).astype(str)

    # Derived fields for analytics
    df['High Purchase Intent'] = np.where(df['Purchase likelihood'] >= 4, 1, 0)
    df['WTP Segment'] = pd.qcut(df['Willingness to pay'], q=3, labels=['Low WTP', 'Medium WTP', 'High WTP'])
    df['Frequency Segment'] = pd.cut(
        df['Purchase Frequency per year'],
        bins=[0, 3, 5, df['Purchase Frequency per year'].max()],
        labels=['Low Frequency', 'Medium Frequency', 'High Frequency'],
        include_lowest=True,
    )

    return df.reset_index(drop=True), sheet_name


def build_classifier(df: pd.DataFrame):
    features = [
        'Age', 'Gender', 'City', 'Monthly Income', 'Preferred Fragrance', 'Shopping Channel',
        'Purchase Frequency per year', 'Willingness to pay', 'Influencer Impact', 'Brand Awareness',
        'Age group', 'Income segment'
    ]
    X = df[features].copy()
    y = df['High Purchase Intent'].copy()

    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols),
        ]
    )

    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced', max_depth=6)
    model = Pipeline([('preprocessor', preprocessor), ('model', clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs),
    }

    fpr, tpr, _ = roc_curve(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    prep = model.named_steps['preprocessor']
    feature_names = prep.get_feature_names_out()
    importances = model.named_steps['model'].feature_importances_
    fi = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(15)

    return metrics, fpr, tpr, cm, fi, X_test.assign(actual=y_test.values, predicted=preds, score=probs)


def build_regression(df: pd.DataFrame):
    features = [
        'Age', 'Gender', 'City', 'Monthly Income', 'Preferred Fragrance', 'Shopping Channel',
        'Purchase Frequency per year', 'Influencer Impact', 'Brand Awareness', 'Purchase likelihood',
        'Age group', 'Income segment'
    ]
    X = df[features].copy()
    y = df['Willingness to pay'].copy()

    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols),
        ]
    )

    reg = RandomForestRegressor(n_estimators=250, random_state=42, max_depth=7)
    model = Pipeline([('preprocessor', preprocessor), ('model', reg)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
    }

    prep = model.named_steps['preprocessor']
    feature_names = prep.get_feature_names_out()
    importances = model.named_steps['model'].feature_importances_
    fi = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(15)

    compare = pd.DataFrame({'Actual': y_test.values, 'Predicted': preds})
    return metrics, fi, compare


def build_clustering(df: pd.DataFrame, chosen_k: int | None = None):
    cluster_cols = [
        'Age', 'Monthly Income', 'Purchase Frequency per year', 'Willingness to pay',
        'Influencer Impact', 'Brand Awareness', 'Purchase likelihood'
    ]
    X = df[cluster_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia_rows = []
    sil_rows = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        inertia_rows.append({'k': k, 'Inertia': model.inertia_})
        sil_rows.append({'k': k, 'Silhouette Score': silhouette_score(X_scaled, labels)})

    sil_df = pd.DataFrame(sil_rows)
    best_k = int(sil_df.loc[sil_df['Silhouette Score'].idxmax(), 'k'])
    final_k = chosen_k or best_k

    final_model = KMeans(n_clusters=final_k, random_state=42, n_init=20)
    final_labels = final_model.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    clustered = df.copy()
    clustered['Cluster'] = final_labels.astype(str)
    clustered['PC1'] = coords[:, 0]
    clustered['PC2'] = coords[:, 1]

    cluster_profile = clustered.groupby('Cluster')[cluster_cols].mean().round(2).reset_index()
    return pd.DataFrame(inertia_rows), sil_df, clustered, cluster_profile, best_k


def build_association_rules(df: pd.DataFrame, min_support: float = 0.15, min_confidence: float = 0.5):
    if not MLXTEND_AVAILABLE:
        raise ImportError('mlxtend is required for association rule mining. Please install requirements.txt before running the app.')
    basket = df[[
        'Gender', 'City', 'Preferred Fragrance', 'Shopping Channel', 'Age group',
        'Income segment', 'WTP Segment', 'Frequency Segment'
    ]].copy()

    basket.columns = [c.replace(' ', '_') for c in basket.columns]
    for col in basket.columns:
        basket[col] = col + '=' + basket[col].astype(str)

    onehot = pd.get_dummies(basket)
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return freq, pd.DataFrame()

    rules = association_rules(freq, metric='confidence', min_threshold=min_confidence)
    if rules.empty:
        return freq, rules

    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
    rules = rules.sort_values(['lift', 'confidence', 'support'], ascending=False)
    return freq.sort_values('support', ascending=False), rules


def make_kpi_card(label: str, value: str):
    st.markdown(
        f"""
        <div style='padding:1rem;border-radius:12px;background:#f6f7fb;border:1px solid #e6e8ef;'>
            <div style='font-size:0.95rem;color:#5b6270;'>{label}</div>
            <div style='font-size:1.8rem;font-weight:700;color:#1f2a44;'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title('💄 Perfume Business Analytics Dashboard')
    st.caption('Built for GitHub + Streamlit deployment using the provided Excel workbook as the data source.')

    uploaded_file = st.sidebar.file_uploader('Optional: upload an updated Excel file', type=['xlsx'])

    try:
        if uploaded_file is not None:
            df, source_sheet = load_business_data(uploaded_file.getvalue(), uploaded_file.name)
            source_name = uploaded_file.name
        else:
            df, source_sheet = load_business_data()
            source_name = find_default_excel().name if find_default_excel() else 'Uploaded workbook'
    except Exception as exc:
        st.error(f'Could not load the dataset: {exc}')
        st.stop()

    st.sidebar.success(f'Data loaded from: {source_name}')
    st.sidebar.info(f'Usable sheet detected: {source_sheet}')
    st.sidebar.write(f'Rows used for modelling: {len(df)}')

    tabs = st.tabs(['Overview', 'Classification', 'Regression', 'Clustering', 'Association Rules', 'Data Explorer'])

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            make_kpi_card('Customers', f'{len(df)}')
        with c2:
            make_kpi_card('Avg WTP', f"₹{df['Willingness to pay'].mean():,.0f}")
        with c3:
            make_kpi_card('High Purchase Intent', f"{df['High Purchase Intent'].mean()*100:.1f}%")
        with c4:
            make_kpi_card('Avg Purchase Frequency', f"{df['Purchase Frequency per year'].mean():.2f}")

        col1, col2 = st.columns(2)
        with col1:
            fragrance_counts = df['Preferred Fragrance'].value_counts().reset_index()
            fragrance_counts.columns = ['Preferred Fragrance', 'Count']
            fig = px.bar(fragrance_counts, x='Preferred Fragrance', y='Count', title='Fragrance Preference Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            channel_counts = df['Shopping Channel'].value_counts().reset_index()
            channel_counts.columns = ['Shopping Channel', 'Count']
            fig = px.pie(channel_counts, names='Shopping Channel', values='Count', title='Shopping Channel Mix')
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            df,
            x='Monthly Income',
            y='Willingness to pay',
            color='Preferred Fragrance',
            size='Purchase Frequency per year',
            hover_data=['Gender', 'City', 'Purchase likelihood'],
            title='Income vs Willingness to Pay'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Business Insight Snapshot')
        st.write(
            'This dashboard supports data-driven product targeting, premium pricing analysis, customer segmentation, '
            'and personalised offer design for the perfume startup idea.'
        )

    with tabs[1]:
        st.subheader('Classification: Predict High Purchase Intent')
        st.write('Target definition used: **High Purchase Intent = Purchase likelihood >= 4**')
        try:
            metrics, fpr, tpr, cm, fi, scored = build_classifier(df)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric('Accuracy', f"{metrics['accuracy']:.3f}")
            m2.metric('Precision', f"{metrics['precision']:.3f}")
            m3.metric('Recall', f"{metrics['recall']:.3f}")
            m4.metric('F1-Score', f"{metrics['f1']:.3f}")
            m5.metric('ROC AUC', f"{metrics['roc_auc']:.3f}")

            col1, col2 = st.columns(2)
            with col1:
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline', line=dict(dash='dash')))
                roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                st.plotly_chart(roc_fig, use_container_width=True)
            with col2:
                cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix')
                cm_fig.update_xaxes(title='Predicted Label')
                cm_fig.update_yaxes(title='Actual Label')
                st.plotly_chart(cm_fig, use_container_width=True)

            fi_fig = px.bar(fi.sort_values('Importance'), x='Importance', y='Feature', orientation='h', title='Top Feature Importance')
            st.plotly_chart(fi_fig, use_container_width=True)
            st.dataframe(scored.head(15), use_container_width=True)
        except Exception as exc:
            st.error(f'Classification section failed: {exc}')

    with tabs[2]:
        st.subheader('Regression: Predict Willingness to Pay')
        try:
            metrics, fi, compare = build_regression(df)
            a, b, c = st.columns(3)
            a.metric('R²', f"{metrics['r2']:.3f}")
            b.metric('MAE', f"₹{metrics['mae']:,.0f}")
            c.metric('RMSE', f"₹{metrics['rmse']:,.0f}")

            col1, col2 = st.columns(2)
            with col1:
                actual_pred_fig = px.scatter(compare, x='Actual', y='Predicted', title='Actual vs Predicted Willingness to Pay')
                actual_pred_fig.add_shape(type='line', x0=compare['Actual'].min(), y0=compare['Actual'].min(),
                                          x1=compare['Actual'].max(), y1=compare['Actual'].max(), line=dict(dash='dash'))
                st.plotly_chart(actual_pred_fig, use_container_width=True)
            with col2:
                fi_fig = px.bar(fi.sort_values('Importance'), x='Importance', y='Feature', orientation='h', title='Regression Feature Importance')
                st.plotly_chart(fi_fig, use_container_width=True)

            st.dataframe(compare.head(20), use_container_width=True)
        except Exception as exc:
            st.error(f'Regression section failed: {exc}')

    with tabs[3]:
        st.subheader('Clustering: Customer Segmentation')
        try:
            selected_k = st.slider('Choose number of clusters', min_value=2, max_value=6, value=3)
            inertia_df, sil_df, clustered, profile, best_k = build_clustering(df, selected_k)
            st.info(f'Silhouette-based suggested k: {best_k}')

            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(inertia_df, x='k', y='Inertia', markers=True, title='Elbow Method')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(sil_df, x='k', y='Silhouette Score', markers=True, title='Silhouette Score by k')
                st.plotly_chart(fig, use_container_width=True)

            cluster_fig = px.scatter(
                clustered,
                x='PC1',
                y='PC2',
                color='Cluster',
                hover_data=['Gender', 'City', 'Preferred Fragrance', 'Shopping Channel', 'Willingness to pay'],
                title='Customer Segments (PCA Projection)'
            )
            st.plotly_chart(cluster_fig, use_container_width=True)
            st.subheader('Cluster Profile')
            st.dataframe(profile, use_container_width=True)
        except Exception as exc:
            st.error(f'Clustering section failed: {exc}')

    with tabs[4]:
        st.subheader('Association Rule Mining')
        try:
            col1, col2 = st.columns(2)
            with col1:
                min_support = st.slider('Minimum support', min_value=0.05, max_value=0.5, value=0.15, step=0.01)
            with col2:
                min_conf = st.slider('Minimum confidence', min_value=0.1, max_value=0.95, value=0.5, step=0.05)

            freq, rules = build_association_rules(df, min_support=min_support, min_confidence=min_conf)
            if freq.empty or rules.empty:
                st.warning('No association rules found for the selected support/confidence settings. Try lowering the thresholds.')
            else:
                top_rules = rules.head(20).copy()
                st.dataframe(top_rules, use_container_width=True)
                fig = px.scatter(
                    top_rules,
                    x='confidence',
                    y='lift',
                    size='support',
                    color='lift',
                    hover_data=['antecedents', 'consequents'],
                    title='Association Rules by Confidence and Lift'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write('Use **confidence** to judge how reliably the rule occurs, and **lift** to judge how strong the relationship is compared with chance.')
        except Exception as exc:
            st.error(f'Association rules section failed: {exc}')

    with tabs[5]:
        st.subheader('Data Explorer')
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label='Download cleaned modelling dataset as CSV',
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='cleaned_modelling_dataset.csv',
            mime='text/csv'
        )


if __name__ == '__main__':
    main()
