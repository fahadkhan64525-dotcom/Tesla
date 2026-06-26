"""
Tesla Stock Price Prediction – Streamlit App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Price Predictor fahad",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #CC0000;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #CC0000;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.4rem 0;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚗 Tesla (TSLA) Stock Price Predictor...Fahad</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning with SimpleRNN & LSTM | Financial Services</div>', unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bb/Tesla_T_symbol.svg", width=60)
    st.title("⚙️ Configuration")

    st.subheader("📁 Data")
    uploaded_file = st.file_uploader("Upload TSLA.csv", type=["csv"])
    st.caption("Download dataset from the project brief link.")

    st.subheader("🧠 Model")
    model_type = st.selectbox("Select Model", ["SimpleRNN", "LSTM", "Both (Compare)"])
    forecast_horizon = st.selectbox("Forecast Horizon", [1, 5, 10], format_func=lambda x: f"{x} Day(s)")
    seq_length = st.slider("Sequence Length (days)", min_value=20, max_value=120, value=60, step=10)

    st.subheader("🔧 Hyperparameters")
    units = st.selectbox("LSTM/RNN Units", [32, 64, 128, 256], index=1)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    epochs = st.slider("Epochs", 10, 150, 50, 10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    train_split = st.slider("Train/Test Split (%)", 60, 90, 80, 5)

    st.subheader("📊 Features")
    use_ma = st.checkbox("Moving Averages (7, 21)", value=True)
    use_rsi = st.checkbox("RSI-14", value=True)
    use_vol = st.checkbox("Volatility (21-day)", value=True)
    use_volume = st.checkbox("Trading Volume", value=True)

    run_btn = st.button("🚀 Train & Predict", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def load_and_clean(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df = df[~df.duplicated()]
    return df


def engineer_features(df, use_ma, use_rsi, use_vol, use_volume):
    feat = df[['Adj Close']].copy()
    if use_ma:
        feat['MA_7']  = feat['Adj Close'].rolling(7).mean()
        feat['MA_21'] = feat['Adj Close'].rolling(21).mean()
    if use_rsi:
        delta = feat['Adj Close'].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        feat['RSI_14'] = 100 - (100 / (1 + rs))
    if use_vol:
        ret = df['Adj Close'].pct_change() * 100
        feat['Volatility'] = ret.rolling(21).std()
    if use_volume:
        feat['Volume'] = df['Volume']
    feat.dropna(inplace=True)
    return feat


def create_sequences(data, target, seq_len, horizon):
    X, y = [], []
    for i in range(seq_len, len(data) - horizon + 1):
        X.append(data[i - seq_len : i])
        y.append(target[i + horizon - 1])
    return np.array(X), np.array(y)


def build_model(model_type, units, dropout, lr, input_shape):
    model = Sequential()
    if model_type == "SimpleRNN":
        model.add(SimpleRNN(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(SimpleRNN(units // 2, return_sequences=False))
        model.add(Dropout(dropout))
    else:  # LSTM
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units // 2, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units // 4, return_sequences=False))
        model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model


def compute_metrics(actual, pred):
    mse  = mean_squared_error(actual, pred)
    return {
        'MSE':  mse,
        'RMSE': np.sqrt(mse),
        'MAE':  mean_absolute_error(actual, pred),
        'R²':   r2_score(actual, pred)
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if uploaded_file is None:
    st.info("👈 Upload **TSLA.csv** in the sidebar to get started.")
    st.markdown("""
    ### 📖 About This App
    This application trains **SimpleRNN** and **LSTM** deep learning models on Tesla's historical
    stock data to forecast closing prices.

    **Features:**
    - Interactive hyperparameter tuning
    - 1-day, 5-day, and 10-day forecast horizons
    - Full EDA with interactive charts
    - Side-by-side model comparison (RMSE, MAE, R²)
    - Downloadable predictions as CSV

    **Steps:**
    1. Upload `TSLA.csv` in the sidebar
    2. Choose your model and hyperparameters
    3. Click **Train & Predict**
    """)
    st.stop()

# ─── Load Data ───────────────────────────────────────────────
df = load_and_clean(uploaded_file)

# ─── Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Training", "📈 Predictions", "📋 Metrics"])

# ══════════════════════════════════
# TAB 1 – EDA
# ══════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows",     f"{len(df):,}")
    col2.metric("Date Range",     f"{df.index.min().year} – {df.index.max().year}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum()}")
    col4.metric("Avg Close",      f"${df['Adj Close'].mean():.2f}")

    st.markdown("---")

    # Closing Price
    st.subheader("Closing Price History")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df.index, df['Adj Close'], linewidth=1.2, color='#CC0000')
    ax1.fill_between(df.index, df['Adj Close'], alpha=0.1, color='#CC0000')
    ax1.set_ylabel("Adj Close (USD)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.set_title("Tesla (TSLA) Adjusted Closing Price")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    # OHLCV summary
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Volume Over Time")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.bar(df.index, df['Volume'], color='steelblue', alpha=0.6, width=1)
        ax2.set_ylabel("Volume")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_b:
        st.subheader("Daily Returns Distribution")
        daily_ret = df['Adj Close'].pct_change() * 100
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.hist(daily_ret.dropna(), bins=80, color='darkorange', edgecolor='white', alpha=0.8)
        ax3.set_xlabel("Daily Return (%)")
        ax3.set_ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # Rolling averages
    st.subheader("Moving Averages")
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(df.index, df['Adj Close'],                       label='Close',    linewidth=0.8, alpha=0.7)
    ax4.plot(df.index, df['Adj Close'].rolling(30).mean(),    label='30-day MA', linewidth=1.8)
    ax4.plot(df.index, df['Adj Close'].rolling(90).mean(),    label='90-day MA', linewidth=1.8)
    ax4.plot(df.index, df['Adj Close'].rolling(200).mean(),   label='200-day MA',linewidth=1.8)
    ax4.legend()
    ax4.set_ylabel("Price (USD)")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.subheader("Raw Data Sample")
    st.dataframe(df.tail(20), use_container_width=True)


# ══════════════════════════════════
# TAB 2 – Training
# ══════════════════════════════════
with tab2:
    if not run_btn:
        st.info("Configure hyperparameters in the sidebar and click **Train & Predict** to begin.")
    else:
        st.subheader("🔧 Preprocessing")

        # Feature engineering
        with st.spinner("Engineering features..."):
            feat_df = engineer_features(df, use_ma, use_rsi, use_vol, use_volume)

        st.success(f"✅ Feature set ready: {feat_df.shape[0]} rows × {feat_df.shape[1]} features")
        st.dataframe(feat_df.tail(10), use_container_width=True)

        # Scale
        scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        scaled = scaler.fit_transform(feat_df)
        target_scaled = target_scaler.fit_transform(feat_df[['Adj Close']])

        # Split
        split = int(len(scaled) * (train_split / 100))
        train_d = scaled[:split]
        test_d  = scaled[split - seq_length:]
        train_t = target_scaled[:split]
        test_t  = target_scaled[split - seq_length:]

        X_tr, y_tr = create_sequences(train_d, train_t, seq_length, forecast_horizon)
        X_te, y_te = create_sequences(test_d,  test_t,  seq_length, forecast_horizon)

        st.write(f"**Train sequences:** {X_tr.shape} | **Test sequences:** {X_te.shape}")

        input_shape = (X_tr.shape[1], X_tr.shape[2])
        models_to_train = ["SimpleRNN", "LSTM"] if model_type == "Both (Compare)" else [model_type]

        st.session_state['results'] = {}
        st.session_state['target_scaler'] = target_scaler
        st.session_state['y_te'] = y_te
        st.session_state['feat_df'] = feat_df
        st.session_state['split'] = split

        for mname in models_to_train:
            st.subheader(f"Training {mname}")
            prog_bar = st.progress(0, text=f"Initialising {mname}...")

            model = build_model(mname, units, dropout_rate, learning_rate, input_shape)

            history_data = {'loss': [], 'val_loss': []}

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pct = int((epoch + 1) / epochs * 100)
                    prog_bar.progress(pct, text=f"Epoch {epoch+1}/{epochs} | loss: {logs['loss']:.6f} | val_loss: {logs['val_loss']:.6f}")
                    history_data['loss'].append(logs['loss'])
                    history_data['val_loss'].append(logs['val_loss'])

            cb = [
                EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=0),
                StreamlitCallback()
            ]

            model.fit(X_tr, y_tr,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_split=0.1,
                      callbacks=cb,
                      verbose=0)

            prog_bar.progress(100, text=f"✅ {mname} training complete!")

            # Predict
            pred_s = model.predict(X_te, verbose=0)
            pred   = target_scaler.inverse_transform(pred_s)
            actual = target_scaler.inverse_transform(y_te.reshape(-1, 1))
            metrics = compute_metrics(actual, pred)

            st.session_state['results'][mname] = {
                'model':   model,
                'pred':    pred,
                'actual':  actual,
                'metrics': metrics,
                'history': history_data
            }

            # Loss curve
            fig_loss, ax_loss = plt.subplots(figsize=(10, 3))
            ax_loss.plot(history_data['loss'],     label='Train Loss')
            ax_loss.plot(history_data['val_loss'], label='Val Loss')
            ax_loss.set_title(f'{mname} Training Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('MSE')
            ax_loss.legend()
            plt.tight_layout()
            st.pyplot(fig_loss)
            plt.close()

            # Quick metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MSE",  f"{metrics['MSE']:.4f}")
            c2.metric("RMSE", f"${metrics['RMSE']:.2f}")
            c3.metric("MAE",  f"${metrics['MAE']:.2f}")
            c4.metric("R²",   f"{metrics['R²']:.4f}")


# ══════════════════════════════════
# TAB 3 – Predictions
# ══════════════════════════════════
with tab3:
    if 'results' not in st.session_state or not st.session_state['results']:
        st.info("Train a model first using the sidebar.")
    else:
        results = st.session_state['results']
        target_scaler = st.session_state['target_scaler']

        for mname, res in results.items():
            st.subheader(f"📈 {mname} – Actual vs Predicted ({forecast_horizon}-day horizon)")
            actual = res['actual']
            pred   = res['pred']
            n = len(actual)

            fig_p, ax_p = plt.subplots(figsize=(13, 5))
            ax_p.plot(range(n), actual, label='Actual',    color='steelblue',  linewidth=2)
            ax_p.plot(range(n), pred,   label='Predicted', color='darkorange', linewidth=1.8, linestyle='--')
            ax_p.fill_between(range(n),
                              actual.flatten(),
                              pred.flatten(),
                              alpha=0.1, color='gray', label='Error Band')
            ax_p.set_title(f'{mname} | {forecast_horizon}-Day Forecast  |  RMSE: ${res["metrics"]["RMSE"]:.2f}  |  R²: {res["metrics"]["R²"]:.4f}')
            ax_p.set_xlabel('Test Trading Days')
            ax_p.set_ylabel('TSLA Price (USD)')
            ax_p.legend()
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close()

            # Download predictions
            out_df = pd.DataFrame({'Actual': actual.flatten(), 'Predicted': pred.flatten()})
            csv = out_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Download {mname} Predictions (CSV)",
                data=csv,
                file_name=f"{mname}_predictions_{forecast_horizon}day.csv",
                mime='text/csv'
            )
            st.markdown("---")

        # Overlay if both models trained
        if len(results) == 2:
            st.subheader("🔀 SimpleRNN vs LSTM Overlay")
            names = list(results.keys())
            r0, r1 = results[names[0]], results[names[1]]
            n = min(len(r0['actual']), len(r1['pred']))

            fig_ov, ax_ov = plt.subplots(figsize=(13, 5))
            ax_ov.plot(range(n), r0['actual'][:n], label='Actual',          color='black',      linewidth=2)
            ax_ov.plot(range(n), r0['pred'][:n],   label=names[0],          color='steelblue',  linewidth=1.5, linestyle='--')
            ax_ov.plot(range(n), r1['pred'][:n],   label=names[1],          color='darkorange', linewidth=1.5, linestyle='--')
            ax_ov.set_title('Model Comparison – Actual vs Predicted')
            ax_ov.set_xlabel('Test Trading Days')
            ax_ov.set_ylabel('TSLA Price (USD)')
            ax_ov.legend()
            plt.tight_layout()
            st.pyplot(fig_ov)
            plt.close()


# ══════════════════════════════════
# TAB 4 – Metrics
# ══════════════════════════════════
with tab4:
    if 'results' not in st.session_state or not st.session_state['results']:
        st.info("Train a model first to see metrics.")
    else:
        results = st.session_state['results']

        st.subheader("📋 Performance Summary")

        rows = []
        for mname, res in results.items():
            row = {'Model': mname}
            row.update(res['metrics'])
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df.style.highlight_min(
            subset=['MSE', 'RMSE', 'MAE'], color='#d4edda'
        ).highlight_max(
            subset=['R²'], color='#d4edda'
        ), use_container_width=True)

        # Bar charts
        st.subheader("📊 Visual Comparison")
        fig_m, axes_m = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes_m, ['RMSE', 'MAE', 'R²']):
            bars = ax.bar(metrics_df['Model'], metrics_df[metric],
                          color=['#4C72B0', '#DD8452'], edgecolor='white', width=0.4)
            ax.set_title(metric, fontsize=12)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f'{bar.get_height():.4f}',
                        ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_m)
        plt.close()

        # Interpretation
        st.subheader("🔍 Interpretation")
        if len(results) == 2:
            names = list(results.keys())
            r0_rmse = results[names[0]]['metrics']['RMSE']
            r1_rmse = results[names[1]]['metrics']['RMSE']
            better = names[0] if r0_rmse < r1_rmse else names[1]
            st.success(f"**{better}** achieved a lower RMSE on the test set for the {forecast_horizon}-day forecast horizon.")

        st.markdown("""
        **Metric Guide:**
        - **MSE** – Mean Squared Error (lower = better; penalises large errors)
        - **RMSE** – Root MSE in original price units (USD)
        - **MAE** – Mean Absolute Error (average dollar error per day)
        - **R²** – Explained variance (closer to 1.0 = better fit)
        """)

        st.subheader("📝 Project Notes")
        st.markdown("""
        **Missing Value Handling:**  
        Forward-fill followed by backward-fill was used instead of mean/median imputation.
        This preserves the temporal structure of the time series and avoids introducing
        statistically "impossible" prices that would distort the sequence model's learning.

        **Model Choice Rationale:**  
        Both SimpleRNN and LSTM belong to the Recurrent Neural Network family, making them
        well-suited for sequential stock data. LSTM's gating mechanism (input, forget, output gates)
        allows it to retain long-term dependencies better than SimpleRNN, which suffers from
        vanishing gradients over long sequences.

        **Limitations:**  
        These models use only technical price features. Real-world trading strategies should
        incorporate fundamental analysis, news sentiment, and macro indicators.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#999;font-size:0.85rem;'>"
    "Tesla Stock Price Prediction · Deep Learning Project · Built with Streamlit & TensorFlow"
    "</p>",
    unsafe_allow_html=True
)
