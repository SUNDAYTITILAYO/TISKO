import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Set page configuration
st.set_page_config(
    page_title="Enhanced Data Analysis App",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = {}

def load_data():
    st.title("📊 Enhanced Data Analysis Platform")
    st.markdown("---")
    
    # File upload with additional information
    st.markdown("### 📁 Data Upload")
    st.markdown("Upload your CSV file to begin the analysis. Make sure your data is clean and properly formatted.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file to analyze"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner('Loading and processing your data...'):
                df = pd.read_csv(uploaded_file)
                st.session_state["data"] = df
                st.success("✅ File uploaded and loaded successfully!")
                
                # Display basic dataset information
                st.markdown("### 📋 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                return True
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return False
    return False

def preprocess_data(data):
    st.markdown("### 🔧 Data Preprocessing")
    
    with st.expander("Preprocessing Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Handle missing values
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["None", "Drop", "Fill with Mean", "Fill with Median", "Fill with Mode"]
            )
            
            # Handle categorical variables
            handle_categorical = st.selectbox(
                "Handle Categorical Variables",
                ["None", "Label Encoding", "Drop Categorical"]
            )
        
        with col2:
            # Feature scaling
            scaling_method = st.selectbox(
                "Feature Scaling",
                ["None", "StandardScaler", "MinMaxScaler"]
            )
            
            # Remove outliers
            remove_outliers = st.checkbox("Remove Outliers (IQR method)")
    
    processed_data = data.copy()
    
    if handle_missing != "None":
        if handle_missing == "Drop":
            processed_data = processed_data.dropna()
        elif handle_missing == "Fill with Mean":
            processed_data = processed_data.fillna(processed_data.mean())
        elif handle_missing == "Fill with Median":
            processed_data = processed_data.fillna(processed_data.median())
        elif handle_missing == "Fill with Mode":
            processed_data = processed_data.fillna(processed_data.mode().iloc[0])
    
    if handle_categorical != "None":
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        if handle_categorical == "Label Encoding":
            le = LabelEncoder()
            for col in categorical_columns:
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
        elif handle_categorical == "Drop Categorical":
            processed_data = processed_data.select_dtypes(exclude=['object'])
    
    if scaling_method != "None":
        numerical_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        processed_data[numerical_columns] = scaler.fit_transform(processed_data[numerical_columns])
    
    if remove_outliers:
        numerical_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns
        for column in numerical_columns:
            Q1 = processed_data[column].quantile(0.25)
            Q3 = processed_data[column].quantile(0.75)
            IQR = Q3 - Q1
            processed_data = processed_data[
                (processed_data[column] >= Q1 - 1.5 * IQR) & 
                (processed_data[column] <= Q3 + 1.5 * IQR)
            ]
    
    st.session_state['processed_data'] = processed_data
    return processed_data

def perform_eda(data):
    st.markdown("### 📊 Exploratory Data Analysis")
    
    with st.expander("EDA Options", expanded=True):
        eda_options = st.multiselect(
            "Select EDA Operations",
            ["Basic Statistics", "Correlation Analysis", "Distribution Analysis", 
             "Missing Values Analysis", "Data Types Information"]
        )
    
    if "Basic Statistics" in eda_options:
        st.subheader("📈 Basic Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Numerical Statistics:")
            st.write(data.describe())
        with col2:
            st.write("Dataset Info:")
            buffer = StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
    
    if "Correlation Analysis" in eda_options:
        st.subheader("🔄 Correlation Analysis")
        numerical_data = data.select_dtypes(include=['float64', 'int64'])
        if not numerical_data.empty:
            fig = px.imshow(
                numerical_data.corr(),
                title="Correlation Heatmap",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if "Distribution Analysis" in eda_options:
        st.subheader("📊 Distribution Analysis")
        column = st.selectbox("Select Column for Distribution", data.columns)
        if data[column].dtype in ['float64', 'int64']:
            fig = px.histogram(
                data,
                x=column,
                title=f"Distribution of {column}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if "Missing Values Analysis" in eda_options:
        st.subheader("❓ Missing Values Analysis")
        missing_data = pd.DataFrame({
            'Column': data.columns,
            'Missing Values': data.isnull().sum(),
            'Percentage': (data.isnull().sum() / len(data) * 100).round(2)
        })
        fig = px.bar(
            missing_data,
            x='Column',
            y='Percentage',
            title="Missing Values Percentage by Column"
        )
        st.plotly_chart(fig, use_container_width=True)

def create_visualizations(data):
    st.markdown("### 📈 Data Visualization")
    
    chart_types = {
        "Scatter Plot": px.scatter,
        "Line Chart": px.line,
        "Bar Chart": px.bar,
        "Box Plot": px.box,
        "Violin Plot": px.violin,
        "Histogram": px.histogram,
        "Word Cloud": "wordcloud"
    }
    
    with st.expander("Visualization Options", expanded=True):
        chart_type = st.selectbox("Select Chart Type", list(chart_types.keys()))
        
        if chart_type != "Word Cloud":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis", data.columns)
            with col2:
                y_axis = st.selectbox("Select Y-axis", data.columns)
            
            color_var = st.selectbox("Color Variable (optional)", ["None"] + list(data.columns))
            
            # Additional customization options
            custom_title = st.text_input("Custom Title (optional)")
            
        else:
            text_column = st.selectbox("Select Text Column", data.select_dtypes(include=['object']).columns)
    
    if chart_type != "Word Cloud":
        fig = create_plotly_chart(data, chart_type, x_axis, y_axis, 
                                color_var if color_var != "None" else None,
                                custom_title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        create_wordcloud(data, text_column)

def create_plotly_chart(data, chart_type, x, y, color=None, title=None):
    chart_fn = chart_types[chart_type]
    
    kwargs = {
        "data_frame": data,
        "x": x,
        "y": y,
        "title": title or f"{chart_type} of {y} vs {x}",
        "template": "plotly_white"
    }
    
    if color:
        kwargs["color"] = color
    
    fig = chart_fn(**kwargs)
    fig.update_layout(
        title_x=0.5,
        margin=dict(t=100),
        height=600
    )
    return fig

def create_wordcloud(data, text_column):
    if text_column:
        text_data = " ".join(data[text_column].astype(str).dropna())
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=STOPWORDS,
            min_font_size=10
        ).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

def build_model(data):
    st.markdown("### 🤖 Machine Learning Modeling")
    
    with st.expander("Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            target = st.selectbox("Select Target Variable", data.columns)
            test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
            
        with col2:
            features = st.multiselect(
                "Select Features",
                [col for col in data.columns if col != target],
                default=[col for col in data.columns if col != target]
            )
            
        model_type = st.selectbox(
            "Select Model",
            ["Linear Regression", "Random Forest", "SVR", "Decision Tree", "KNN"]
        )
        
        # Advanced model parameters
        with st.expander("Advanced Model Parameters"):
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 200, 100)
                max_depth = st.slider("Max Depth", 1, 20, 10)
            elif model_type == "KNN":
                n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
            elif model_type == "SVR":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            X = data[features]
            y = data[target]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Model selection and training
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            elif model_type == "SVR":
                model = SVR(kernel=kernel)
            elif model_type == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            elif model_type == "KNN":
                model = KNeighborsRegressor(n_neighbors=n_neighbors)
            
            # Model training and prediction
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            # Store results
            st.session_state['model_results'] = {
                'predictions': predictions,
                'y_test': y_test,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_scores': cv_scores
            }
            
            # Display results
            display_model_results()

def display_model_results():
    results = st.session_state['model_results']
    
    st.markdown("### 📊 Model Performance")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{results['r2']:.3f}")
    with col2:
        st.metric("RMSE", f"{results['rmse']:.3f}")
    with col3:
        st.metric("MAE", f"{results['mae']:.3f}")
   # with col4:
