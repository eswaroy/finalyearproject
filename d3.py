import os
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, ks_2samp
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------------- Parameters -------------------
SEQ_LENGTH = 24
LATENT_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 100
FEATURES = ['HR', 'Temp', 'RespRate', 'DiasABP', 'Glucose', 'BUN', 'Creatinine', 'WBC', 'HCT', 'GCS']
TABULAR_FEATURES = ['Age', 'Gender', 'Height', 'Weight', 'ICUType', 'Outcome']
COND_FEATURES = ['Age', 'Gender', 'disease_label']

# ------------------- Data Loading & Preprocessing -------------------
def load_and_preprocess_data(time_series_path="cleaned_merged_data.csv", tabular_path=r"C:\Users\Asus\Downloads\syn\physionet_output\cleaned_tabular_data.csv"):
    """
    Load and preprocess time series and tabular data for model training.
    
    Args:
        time_series_path: Path to CSV file containing time series data
        tabular_path: Path to CSV file containing tabular data
        
    Returns:
        tuple: (merged_data, scalers, label_encoders) or (None, None, None) if data loading fails
    """
    try:
        # Try importing streamlit only if needed
        try:
            import streamlit as st
        except ImportError:
            st = None  # Streamlit not needed for CLI mode
        
        # Load data with better error handling
        try:
            time_series_data = pd.read_csv(time_series_path)
            if time_series_data.empty:
                raise ValueError("Time series data file is empty")
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            error_msg = f"Error loading time series data: {str(e)}"
            if st:
                st.error(error_msg)
            print(error_msg)
            return None, None, None
        
        try:
            tabular_data = pd.read_csv(tabular_path)
            if tabular_data.empty:
                raise ValueError("Tabular data file is empty")
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            error_msg = f"Error loading tabular data: {str(e)}"
            if st:
                st.error(error_msg)
            print(error_msg)
            return None, None, None
        
        # Validate required columns
        required_columns = ['RecordID']
        for dataset, name in [(time_series_data, 'time series'), (tabular_data, 'tabular')]:
            missing = [col for col in required_columns if col not in dataset.columns]
            if missing:
                error_msg = f"Missing required columns in {name} data: {missing}"
                if st:
                    st.error(error_msg)
                print(error_msg)
                return None, None, None
        
        # Check for duplicate RecordIDs in tabular data
        tabular_counts = tabular_data['RecordID'].value_counts()
        duplicates = tabular_counts[tabular_counts > 1]
        if not duplicates.empty:
            warning_msg = f"Warning: {len(duplicates)} RecordIDs in {tabular_path} have multiple rows"
            if st:
                st.warning(warning_msg)
            print(warning_msg)
        
        # Log missing RecordIDs
        missing_in_tabular = set(time_series_data['RecordID']) - set(tabular_data['RecordID'])
        missing_in_time_series = set(tabular_data['RecordID']) - set(time_series_data['RecordID'])
        
        if missing_in_tabular:
            info_msg = f"{len(missing_in_tabular)} RecordIDs in time-series data not found in tabular data"
            if st:
                st.info(info_msg)
            print(info_msg)
            with open("missing_record_ids.txt", "w") as f:
                f.write(f"Missing RecordIDs in tabular data: {missing_in_tabular}\n")
        
        if missing_in_time_series:
            info_msg = f"{len(missing_in_time_series)} RecordIDs in tabular data not found in time-series data"
            if st:
                st.info(info_msg)
            print(info_msg)
            with open("missing_record_ids.txt", "a") as f:
                f.write(f"Missing RecordIDs in time-series data: {missing_in_time_series}\n")
        
        # Merge data and ensure we have at least some records
        merged_data = time_series_data.merge(tabular_data, on='RecordID', how='inner')
        if merged_data.empty:
            error_msg = "No matching records found between time series and tabular data"
            if st:
                st.error(error_msg)
            print(error_msg)
            return None, None, None
        
        # Apply label encoding
        label_encoders = {}
        categorical_columns = ['disease_label', 'Gender', 'ICUType', 'Outcome']
        for col in categorical_columns:
            if col in merged_data.columns:
                label_encoders[col] = LabelEncoder()
                merged_data[col] = label_encoders[col].fit_transform(merged_data[col].astype(str))
        
        # Apply scaling
        scalers = {}
        numeric_columns = FEATURES + ['Age', 'Height', 'Weight']
        for col in numeric_columns:
            if col in merged_data.columns:
                scalers[col] = MinMaxScaler()
                # Handle NaN values before scaling
                if merged_data[col].isna().any():
                    print(f"Warning: NaN values found in column {col}. Filling with mean.")
                    merged_data[col] = merged_data[col].fillna(merged_data[col].mean())
                merged_data[col] = scalers[col].fit_transform(merged_data[[col]])
        
        return merged_data, scalers, label_encoders
    
    except Exception as e:
        error_msg = f"Unexpected error in data preprocessing: {str(e)}"
        if st:
            st.error(error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, None

def preprocess_for_model(data):
    """
    Preprocess data for model training by creating sequences, tabular data, and condition arrays.
    
    Args:
        data: DataFrame containing merged time series and tabular data
        
    Returns:
        tuple: (time_series_tensor, tabular_tensor, conditions_tensor)
    """
    # Check if data is empty
    if data.empty:
        print("Error: Empty data provided to preprocess_for_model")
        return torch.zeros((0, SEQ_LENGTH, len(FEATURES))), torch.zeros((0, len(TABULAR_FEATURES))), torch.zeros((0, len(COND_FEATURES)))
    
    # Check for required columns
    missing_features = [col for col in FEATURES if col not in data.columns]
    missing_tabular = [col for col in TABULAR_FEATURES if col not in data.columns]
    missing_cond = [col for col in COND_FEATURES if col not in data.columns]
    
    if missing_features or missing_tabular or missing_cond:
        print(f"Warning: Missing columns in data: {missing_features + missing_tabular + missing_cond}")
        # Handle missing columns by filling with zeros
        for col in missing_features + missing_tabular + missing_cond:
            data[col] = 0
    
    # Pre-allocate arrays for better memory efficiency
    unique_records = data['RecordID'].unique()
    num_records = len(unique_records)
    
    sequences = np.zeros((num_records, SEQ_LENGTH, len(FEATURES)))
    tabular_data = np.zeros((num_records, len(TABULAR_FEATURES)))
    conds = np.zeros((num_records, len(COND_FEATURES)))
    
    # Create a mapping from RecordID to index for fast lookup
    record_to_idx = {rid: i for i, rid in enumerate(unique_records)}
    
    # Process each record efficiently
    for rid, group in data.groupby('RecordID'):
        idx = record_to_idx[rid]
        
        # Extract time series data
        seq = group[FEATURES].values
        seq_len = min(SEQ_LENGTH, len(seq))
        sequences[idx, :seq_len] = seq[:seq_len]
        
        # Extract tabular and condition data (first row only)
        first_row = group.iloc[0]
        tabular_data[idx] = first_row[TABULAR_FEATURES].values
        conds[idx] = first_row[COND_FEATURES].values
    
    # Convert to PyTorch tensors
    return (torch.from_numpy(sequences).float(),
            torch.from_numpy(tabular_data).float(),
            torch.from_numpy(conds).float())

# ------------------- Models -------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return x * weights

class TabularGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + len(COND_FEATURES), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(TABULAR_FEATURES)),
            nn.Sigmoid()
        )
    
    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.model(x)

class TimeSeriesGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + len(COND_FEATURES), 256),
            nn.ReLU(),
            nn.Linear(256, SEQ_LENGTH * HIDDEN_DIM),
            nn.ReLU()
        )
        self.attn = Attention(HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, len(FEATURES))

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x).view(-1, SEQ_LENGTH, HIDDEN_DIM)
        x = self.attn(x)
        return torch.sigmoid(self.out(x))

class TabularDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(len(TABULAR_FEATURES) + len(COND_FEATURES), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)

class TimeSeriesDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(FEATURES), hidden_size=HIDDEN_DIM, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM + len(COND_FEATURES), 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        output, (h_n, _) = self.lstm(x)
        x = torch.cat([h_n.squeeze(0), cond], dim=1)
        return self.fc(x)

class CrossModalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.tab_to_ts = nn.Sequential(
            nn.Linear(len(TABULAR_FEATURES) + len(COND_FEATURES), 256),
            nn.ReLU(),
            nn.Linear(256, SEQ_LENGTH * len(FEATURES)),
            nn.Sigmoid()
        )
        
        self.ts_encoder = nn.LSTM(len(FEATURES), HIDDEN_DIM, batch_first=True)
        self.ts_to_tab = nn.Sequential(
            nn.Linear(HIDDEN_DIM + len(COND_FEATURES), 128),
            nn.ReLU(),
            nn.Linear(128, len(TABULAR_FEATURES)),
            nn.Sigmoid()
        )
    
    def generate_ts_from_tab(self, tab, cond):
        x = torch.cat([tab, cond], dim=1)
        return self.tab_to_ts(x).view(-1, SEQ_LENGTH, len(FEATURES))
    
    def generate_tab_from_ts(self, ts, cond):
        _, (h_n, _) = self.ts_encoder(ts)
        x = torch.cat([h_n.squeeze(0), cond], dim=1)
        return self.ts_to_tab(x)

# ------------------- Validation Functions -------------------
def calculate_statistical_metrics(real_data, synthetic_data, feature_names):
    metrics = {}
    
    for i, feature in enumerate(feature_names):
        real_col = real_data[:, i] if len(real_data.shape) == 2 else real_data[:, :, i].flatten()
        synth_col = synthetic_data[:, i] if len(synthetic_data.shape) == 2 else synthetic_data[:, :, i].flatten()
        
        w_dist = wasserstein_distance(real_col, synth_col)
        ks_stat, _ = ks_2samp(real_col, synth_col)
        
        metrics[feature] = {
            'wasserstein': w_dist,
            'ks_stat': ks_stat
        }
    
    metrics['overall'] = {
        'avg_wasserstein': np.mean([metrics[f]['wasserstein'] for f in feature_names]),
        'avg_ks_stat': np.mean([metrics[f]['ks_stat'] for f in feature_names])
    }
    
    return metrics

def evaluate_utility(real_ts, real_tab, synth_ts, synth_tab, real_cond, task_type='classification'):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    real_ts_flat = real_ts.reshape(real_ts.shape[0], -1)
    synth_ts_flat = synth_ts.reshape(synth_ts.shape[0], -1)
    
    real_combined = np.hstack([real_ts_flat, real_tab])
    synth_combined = np.hstack([synth_ts_flat, synth_tab])
    
    if task_type == 'classification':
        target = real_cond[:, 2]
        
        X_train, X_test, y_train, y_test = train_test_split(real_combined, target, test_size=0.2)
        real_model = RandomForestClassifier()
        real_model.fit(X_train, y_train)
        real_score = real_model.score(X_test, y_test)
        
        synth_model = RandomForestClassifier()
        synth_model.fit(synth_combined, target)
        synth_score = synth_model.score(X_test, y_test)
        
        return {
            'real_accuracy': real_score,
            'synth_accuracy': synth_score,
            'utility_score': synth_score / real_score if real_score > 0 else 0
        }
    
    else:
        target = real_cond[:, 0]
        
        X_train, X_test, y_train, y_test = train_test_split(real_combined, target, test_size=0.2)
        real_model = RandomForestRegressor()
        real_model.fit(X_train, y_train)
        real_mse = mean_squared_error(y_test, real_model.predict(X_test))
        
        synth_model = RandomForestRegressor()
        synth_model.fit(synth_combined, target)
        synth_mse = mean_squared_error(y_test, synth_model.predict(X_test))
        
        return {
            'real_mse': real_mse,
            'synth_mse': synth_mse,
            'utility_score': real_mse / synth_mse if synth_mse > 0 else 0
        }

# ------------------- Training -------------------
def train_model(merged_data):
    """
    Train the generative models on the provided data.
    
    Args:
        merged_data: DataFrame containing preprocessed data
        
    Returns:
        dict: Training history metrics
    """
    import sys
    
    # Preprocess data
    time_series, tabular, conditions = preprocess_for_model(merged_data)
    
    # Validate data dimensions
    if time_series.size == 0 or tabular.size == 0 or conditions.size == 0:
        print("Error: Empty tensors after preprocessing. Aborting training.")
        return {'ts_gen_loss': [], 'ts_dis_loss': [], 'tab_gen_loss': [], 'tab_dis_loss': [], 'cross_loss': []}
    
    # Create dataset and dataloader with appropriate batch size
    dataset = TensorDataset(time_series, tabular, conditions)
    print(f"Dataset size: {len(dataset)}")
    
    # Adjust batch size if needed
    effective_batch_size = min(BATCH_SIZE, len(dataset))
    if effective_batch_size < BATCH_SIZE:
        print(f"Warning: Dataset size ({len(dataset)}) is smaller than requested batch size ({BATCH_SIZE}).")
        print(f"Using batch size of {effective_batch_size} instead.")
    
    # Create dataloader with appropriate settings
    data_loader = DataLoader(
        dataset, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        drop_last=False,  # Don't drop last batch even if it's smaller
        num_workers=0,    # No extra workers for small datasets
        pin_memory=True   # Speed up data transfer to GPU
    )
    
    print(f"DataLoader created with {len(data_loader)} batches")
    
    # Setup device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    ts_generator = TimeSeriesGenerator().to(device)
    tab_generator = TabularGenerator().to(device)
    cross_modal_generator = CrossModalGenerator().to(device)
    ts_discriminator = TimeSeriesDiscriminator().to(device)
    tab_discriminator = TabularDiscriminator().to(device)
    
    # Make models available globally and in session state if using Streamlit
    try:
        import streamlit as st
        if 'streamlit' in sys.modules:
            st.session_state.ts_generator = ts_generator
            st.session_state.tab_generator = tab_generator
            st.session_state.cross_modal_generator = cross_modal_generator
    except:
        pass
    
    # Make models available in global scope
    globals()['ts_generator'] = ts_generator
    globals()['tab_generator'] = tab_generator
    globals()['cross_modal_generator'] = cross_modal_generator
    
    # Check if model parameters are trainable
    for name, model in [
        ('TS Generator', ts_generator), 
        ('Tab Generator', tab_generator),
        ('Cross-Modal Generator', cross_modal_generator),
        ('TS Discriminator', ts_discriminator),
        ('Tab Discriminator', tab_discriminator)
    ]:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name} has {trainable_params} trainable parameters")
    
    # Initialize optimizers with appropriate learning rates
    optimizer_ts_g = torch.optim.Adam(ts_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_tab_g = torch.optim.Adam(tab_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_cross = torch.optim.Adam(cross_modal_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_ts_d = torch.optim.Adam(ts_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_tab_d = torch.optim.Adam(tab_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss functions
    criterion = nn.BCELoss()
    cross_criterion = nn.MSELoss()
    
    # Initialize metrics history
    metrics_history = {
        'ts_gen_loss': [], 'ts_dis_loss': [],
        'tab_gen_loss': [], 'tab_dis_loss': [],
        'cross_loss': []
    }
    
    # Training loop with better progress tracking
    for epoch in range(EPOCHS):
        # Reset epoch metrics
        epoch_metrics = {
            'ts_gen_loss': 0, 'ts_dis_loss': 0,
            'tab_gen_loss': 0, 'tab_dis_loss': 0,
            'cross_loss': 0
        }
        
        batch_count = 0
        # Show progress using tqdm if available
        try:
            from tqdm import tqdm
            data_iterator = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        except ImportError:
            data_iterator = data_loader
            print(f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (real_ts, real_tab, cond) in enumerate(data_iterator):
            real_ts, real_tab, cond = real_ts.to(device), real_tab.to(device), cond.to(device)
            batch_size = real_ts.size(0)
            batch_count += 1
            
            # Create noise vector
            z = torch.randn(batch_size, LATENT_DIM, device=device)
            
            # ================= Train Discriminators =================
            # Train time series discriminator
            optimizer_ts_d.zero_grad()
            
            # Real samples
            real_ts_pred = ts_discriminator(real_ts, cond)
            real_ts_loss = criterion(real_ts_pred, torch.ones_like(real_ts_pred))
            
            # Fake samples
            with torch.no_grad():  # Don't compute gradients for generator when training discriminator
                fake_ts = ts_generator(z, cond)
            fake_ts_pred = ts_discriminator(fake_ts.detach(), cond)  # Detach to ensure no gradients
            fake_ts_loss = criterion(fake_ts_pred, torch.zeros_like(fake_ts_pred))
            
            # Combined loss
            ts_d_loss = (real_ts_loss + fake_ts_loss) / 2
            ts_d_loss.backward()
            optimizer_ts_d.step()
            
            # Train tabular discriminator
            optimizer_tab_d.zero_grad()
            
            # Real samples
            real_tab_pred = tab_discriminator(real_tab, cond)
            real_tab_loss = criterion(real_tab_pred, torch.ones_like(real_tab_pred))
            
            # Fake samples
            with torch.no_grad():  # Don't compute gradients for generator when training discriminator
                fake_tab = tab_generator(z, cond)
            fake_tab_pred = tab_discriminator(fake_tab.detach(), cond)  # Detach to ensure no gradients
            fake_tab_loss = criterion(fake_tab_pred, torch.zeros_like(fake_tab_pred))
            
            # Combined loss
            tab_d_loss = (real_tab_loss + fake_tab_loss) / 2
            tab_d_loss.backward()
            optimizer_tab_d.step()
            
            # ================= Train Generators =================
            # Train time series generator
            optimizer_ts_g.zero_grad()
            fake_ts = ts_generator(z, cond)
            fake_ts_pred = ts_discriminator(fake_ts, cond)
            ts_g_loss = criterion(fake_ts_pred, torch.ones_like(fake_ts_pred))
            ts_g_loss.backward()
            optimizer_ts_g.step()
            
            # Train tabular generator
            optimizer_tab_g.zero_grad()
            fake_tab = tab_generator(z, cond)
            fake_tab_pred = tab_discriminator(fake_tab, cond)
            tab_g_loss = criterion(fake_tab_pred, torch.ones_like(fake_tab_pred))
            tab_g_loss.backward()
            optimizer_tab_g.step()
            
            # ================= Train Cross-Modal Generator =================
            optimizer_cross.zero_grad()
            
            # Time series to tabular
            generated_tab = cross_modal_generator.generate_tab_from_ts(real_ts, cond)
            tab_recon_loss = cross_criterion(generated_tab, real_tab)
            
            # Tabular to time series
            generated_ts = cross_modal_generator.generate_ts_from_tab(real_tab, cond)
            ts_recon_loss = cross_criterion(generated_ts, real_ts)
            
            # Combined loss
            cross_loss = ts_recon_loss + tab_recon_loss
            cross_loss.backward()
            optimizer_cross.step()
            
            # Update metrics
            epoch_metrics['ts_gen_loss'] += ts_g_loss.item()
            epoch_metrics['ts_dis_loss'] += ts_d_loss.item()
            epoch_metrics['tab_gen_loss'] += tab_g_loss.item()
            epoch_metrics['tab_dis_loss'] += tab_d_loss.item()
            epoch_metrics['cross_loss'] += cross_loss.item()
            
            # Print progress periodically
            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"Batch {batch_idx}/{len(data_loader)} - "
                      f"TS G/D: {ts_g_loss.item():.4f}/{ts_d_loss.item():.4f}, "
                      f"Tab G/D: {tab_g_loss.item():.4f}/{tab_d_loss.item():.4f}, "
                      f"Cross: {cross_loss.item():.4f}")
        
        # Calculate average metrics for this epoch
        for key in epoch_metrics:
            if batch_count > 0:  # Prevent division by zero
                epoch_metrics[key] /= batch_count
            metrics_history[key].append(epoch_metrics[key])
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"TS Gen/Dis Loss: {epoch_metrics['ts_gen_loss']:.4f}/{epoch_metrics['ts_dis_loss']:.4f}")
        print(f"Tab Gen/Dis Loss: {epoch_metrics['tab_gen_loss']:.4f}/{epoch_metrics['tab_dis_loss']:.4f}")
        print(f"Cross Loss: {epoch_metrics['cross_loss']:.4f}")
        print("-" * 50)
    
    # Set models to evaluation mode when done
    ts_generator.eval()
    tab_generator.eval()
    cross_modal_generator.eval()
    
    # Update session state with metrics history if using Streamlit
    try:
        import streamlit as st
        if 'streamlit' in sys.modules:
            st.session_state.metrics_history = metrics_history
    except:
        pass
    
    return metrics_history
def generate_synthetic_data(num_samples, conditions=None, device=None):
    """
    Generate synthetic healthcare data using trained models.
    
    Args:
        num_samples: Number of synthetic samples to generate
        conditions: Optional tensor of conditions (age, gender, disease) for generation
        device: Device to use for generation (GPU/CPU)
        
    Returns:
        tuple: (synthetic_ts, synthetic_tab, conditions_np)
    """
    import sys
    
    # Input validation
    if num_samples <= 0:
        raise ValueError("Number of samples must be positive")
    
    # Handle Streamlit session state vs globals
    try:
        import streamlit as st
        in_streamlit = 'streamlit' in sys.modules
        
        # Get models from appropriate source
        if in_streamlit and 'ts_generator' in st.session_state and 'tab_generator' in st.session_state:
            ts_generator = st.session_state.ts_generator
            tab_generator = st.session_state.tab_generator
        elif all(m in globals() for m in ['ts_generator', 'tab_generator']):
            ts_generator = globals()['ts_generator']
            tab_generator = globals()['tab_generator']
        else:
            raise RuntimeError("Models not found. Please train models first.")
    except:
        # Fallback to global models
        if not all(m in globals() for m in ['ts_generator', 'tab_generator']):
            raise RuntimeError("Models not found. Please train models first.")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure models are in eval mode
    ts_generator.eval()
    tab_generator.eval()
    
    # Generate conditions if not provided
    if conditions is None:
        conditions = torch.zeros(num_samples, len(COND_FEATURES), device=device)
        # Age (normalized to 0-1 range)
        conditions[:, 0] = torch.rand(num_samples, device=device) * 0.9 + 0.05
        # Gender (binary)
        conditions[:, 1] = torch.randint(0, 2, (num_samples,), device=device).float()
        # Disease label (categorical)
        conditions[:, 2] = torch.randint(0, 5, (num_samples,), device=device).float()
    else:
        # Validate provided conditions
        if not isinstance(conditions, torch.Tensor):
            raise TypeError("Conditions must be a PyTorch tensor")
        if conditions.shape[0] != num_samples:
            raise ValueError(f"Expected {num_samples} conditions, got {conditions.shape[0]}")
        if conditions.shape[1] != len(COND_FEATURES):
            raise ValueError(f"Expected {len(COND_FEATURES)} condition features, got {conditions.shape[1]}")
        # Move to correct device if needed
        if conditions.device != device:
            conditions = conditions.to(device)
    
    # Generate latent vectors
    z = torch.randn(num_samples, LATENT_DIM, device=device)
    
    # Generate synthetic data in batches to avoid memory issues
    batch_size = min(32, num_samples)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    synthetic_ts_list = []
    synthetic_tab_list = []
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_z = z[start_idx:end_idx]
            batch_cond = conditions[start_idx:end_idx]
            
            batch_ts = ts_generator(batch_z, batch_cond)
            batch_tab = tab_generator(batch_z, batch_cond)
            
            synthetic_ts_list.append(batch_ts.cpu())
            synthetic_tab_list.append(batch_tab.cpu())
    
    # Concatenate batches
    synthetic_ts = torch.cat(synthetic_ts_list, dim=0).numpy()
    synthetic_tab = torch.cat(synthetic_tab_list, dim=0).numpy()
    conditions_np = conditions.cpu().numpy()
    
    return synthetic_ts, synthetic_tab, conditions_np

def save_synthetic_data(synthetic_ts, synthetic_tab, conditions, scalers, label_encoders, num_samples):
    """
    Save generated synthetic data to CSV files.
    
    Args:
        synthetic_ts: Generated time series data
        synthetic_tab: Generated tabular data
        conditions: Conditions used for generation
        scalers: Dictionary of fitted MinMaxScalers
        label_encoders: Dictionary of fitted LabelEncoders
        num_samples: Number of samples generated
        
    Returns:
        tuple: (ts_file_path, tab_file_path)
    """
    # Create output directory if it doesn't exist
    output_dir = "synthetic_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize data structures
    ts_data = []
    tab_data = []
    
    # Process time series data
    for i in range(synthetic_ts.shape[0]):
        for j in range(synthetic_ts.shape[1]):
            record = {
                'RecordID': i + 1,
                'TimeStep': j
            }
            
            # Process each feature
            for k, feature in enumerate(FEATURES):
                try:
                    if feature in scalers and scalers[feature] is not None:
                        # Clip values to valid range for inverse transform
                        val = np.clip(synthetic_ts[i, j, k], 0, 1)
                        record[feature] = scalers[feature].inverse_transform([[val]])[0][0]
                    else:
                        record[feature] = synthetic_ts[i, j, k]
                except Exception as e:
                    print(f"Error processing feature {feature}: {str(e)}")
                    record[feature] = synthetic_ts[i, j, k]  # Fallback to raw value
            
            ts_data.append(record)
    
    # Process tabular data
    for i in range(synthetic_tab.shape[0]):
        record = {'RecordID': i + 1}
        
        # Process tabular features
        for j, feature in enumerate(TABULAR_FEATURES):
            try:
                if feature in scalers and scalers[feature] is not None:
                    # Clip values to valid range for inverse transform
                    val = np.clip(synthetic_tab[i, j], 0, 1)
                    record[feature] = scalers[feature].inverse_transform([[val]])[0][0]
                elif feature in label_encoders and label_encoders[feature] is not None:
                    # Handle categorical features
                    num_classes = len(label_encoders[feature].classes_)
                    value = int(round(synthetic_tab[i, j] * (num_classes - 1)))
                    value = max(0, min(value, num_classes - 1))  # Ensure valid class index
                    record[feature] = label_encoders[feature].inverse_transform([value])[0]
                else:
                    record[feature] = synthetic_tab[i, j]
            except Exception as e:
                print(f"Error processing tabular feature {feature}: {str(e)}")
                record[feature] = synthetic_tab[i, j]  # Fallback to raw value
        
        # Process condition features
        for j, feature in enumerate(COND_FEATURES):
            try:
                if feature in scalers and scalers[feature] is not None:
                    # Clip values to valid range for inverse transform
                    val = np.clip(conditions[i, j], 0, 1)
                    record[feature] = scalers[feature].inverse_transform([[val]])[0][0]
                elif feature in label_encoders and feature != 'disease_label' and label_encoders[feature] is not None:
                    # Handle categorical features
                    value = int(round(conditions[i, j]))
                    num_classes = len(label_encoders[feature].classes_)
                    value = max(0, min(value, num_classes - 1))  # Ensure valid class index
                    record[feature] = label_encoders[feature].inverse_transform([value])[0]
                elif feature == 'disease_label':
                    value = int(round(conditions[i, j]))
                    record[feature] = value
                else:
                    record[feature] = conditions[i, j]
            except Exception as e:
                print(f"Error processing condition feature {feature}: {str(e)}")
                record[feature] = conditions[i, j]  # Fallback to raw value
                
        tab_data.append(record)
    
    # Create DataFrames
    try:
        ts_df = pd.DataFrame(ts_data)
        tab_df = pd.DataFrame(tab_data)
    except Exception as e:
        print(f"Error creating DataFrames: {str(e)}")
        return None, None
    
    # Define file paths
    ts_file = os.path.join(output_dir, f"synthetic_timeseries_{timestamp}_{num_samples}.csv")
    tab_file = os.path.join(output_dir, f"synthetic_tabular_{timestamp}_{num_samples}.csv")
    
    # Save to CSV files with error handling
    try:
        ts_df.to_csv(ts_file, index=False)
        tab_df.to_csv(tab_file, index=False)
        print(f"Successfully saved synthetic data to {ts_file} and {tab_file}")
        return ts_file, tab_file
    except Exception as e:
        print(f"Error saving synthetic data: {str(e)}")
        return None, None
def dashboard():
    import streamlit as st
    import streamlit.config
    os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"  # Disable Streamlit file watcher
    streamlit.config.set_option("server.fileWatcherType", "none")
    
    st.set_page_config(page_title="Synthetic Healthcare Data Generator", layout="wide")
    
    # Improved session state initialization - check for all required models
    if 'ts_generator' in st.session_state and 'tab_generator' in st.session_state and 'cross_modal_generator' in st.session_state:
        st.session_state.models_initialized = True
    else:
        st.session_state.models_initialized = False
    
    st.title("Multimodal Synthetic Healthcare Data Generator")
    
    tabs = st.tabs(["Data Generation", "Validation", "Training", "About"])
    
    with tabs[0]:  # DATA GENERATION TAB
        st.header("Generate Synthetic Healthcare Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Characteristics")
            age = st.slider("Age", 18, 90, 40)
            gender = st.selectbox("Gender", ["Male", "Female"])
            disease = st.selectbox("Disease Type", ["Diabetes", "Heart Disease", "Respiratory", "Neurological", "Other"])
            
            gender_encoded = 0 if gender == "Male" else 1
            disease_encoded = {"Diabetes": 0, "Heart Disease": 1, "Respiratory": 2, "Neurological": 3, "Other": 4}[disease]
            
            num_records = st.number_input("Number of Records to Generate", min_value=1, max_value=1000, value=10)
            
            generate_button = st.button("Generate Data")
        
        with col2:
            st.subheader("Generation Settings")
            st.write("Configure patient characteristics and number of records to generate synthetic data.")
        
        if generate_button:
            if not st.session_state.models_initialized:
                st.error("Models not trained. Please train the models first in the Training tab.")
            else:
                with st.spinner("Generating synthetic data..."):
                    # Get models from session state
                    ts_generator = st.session_state.ts_generator
                    tab_generator = st.session_state.tab_generator
                    cross_modal_generator = st.session_state.cross_modal_generator
                    
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    conditions = torch.zeros(num_records, len(COND_FEATURES)).to(device)
                    conditions[:, 0] = torch.tensor([age / 90] * num_records)
                    conditions[:, 1] = torch.tensor([gender_encoded] * num_records).float()
                    conditions[:, 2] = torch.tensor([disease_encoded] * num_records).float()
                    
                    try:
                        synthetic_ts, synthetic_tab, conditions_np = generate_synthetic_data(num_records, conditions, device)
                        
                        # Get scalers and label encoders from session state
                        scalers = st.session_state.get('scalers', {})
                        label_encoders = st.session_state.get('label_encoders', {})
                        
                        ts_file, tab_file = save_synthetic_data(
                            synthetic_ts, synthetic_tab, conditions_np, 
                            scalers, label_encoders, 
                            num_records
                        )
                        st.success(f"Successfully generated {num_records} synthetic patient records!")
                        st.info(f"Saved time series data to: {ts_file}")
                        st.info(f"Saved tabular data to: {tab_file}")
                        
                        st.subheader("Sample Generated Data")
                        sample_idx = 0
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Time Series Data (First Patient)")
                            df_ts = pd.DataFrame(synthetic_ts[sample_idx], columns=FEATURES)
                            st.dataframe(df_ts)
                            
                            fig = px.line(df_ts)
                            fig.update_layout(title="Time Series Visualization", xaxis_title="Time Step", yaxis_title="Value")
                            st.plotly_chart(fig)
                        
                        with col2:
                            st.write("Tabular Data")
                            df_tab = pd.DataFrame([synthetic_tab[sample_idx]], columns=TABULAR_FEATURES)
                            st.dataframe(df_tab)
                    
                    except Exception as e:
                        st.error(f"Error generating data: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    
    # Validation tab remains mostly unchanged, just need to check st.session_state
    # ...
    
    with tabs[2]:  # TRAINING TAB
        st.header("Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Parameters")
            
            train_epochs = st.slider("Number of Epochs", 10, 300, 100)
            train_batch_size = st.slider("Batch Size", 8, 128, 32)
            
            train_button = st.button("Train Models")
        
        with col2:
            st.subheader("Training Status")
            
            if 'metrics_history' in st.session_state:
                metrics_history = st.session_state.metrics_history
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=metrics_history['ts_gen_loss'], name="TS Generator Loss"))
                fig.add_trace(go.Scatter(y=metrics_history['ts_dis_loss'], name="TS Discriminator Loss"))
                fig.add_trace(go.Scatter(y=metrics_history['tab_gen_loss'], name="Tab Generator Loss"))
                fig.add_trace(go.Scatter(y=metrics_history['tab_dis_loss'], name="Tab Discriminator Loss"))
                fig.add_trace(go.Scatter(y=metrics_history['cross_loss'], name="Cross-Modal Loss"))
                fig.update_layout(title="Training Losses", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig)
                
                st.success("Models are trained and ready to use!")
        
        if train_button:
            global EPOCHS, BATCH_SIZE
            EPOCHS = train_epochs
            BATCH_SIZE = train_batch_size
            
            with st.spinner("Loading and preprocessing data..."):
                merged_data, scalers, label_encoders = load_and_preprocess_data()
                
                if merged_data is not None:
                    # Store in session state
                    st.session_state.scalers = scalers
                    st.session_state.label_encoders = label_encoders
                    
                    with st.spinner(f"Training models for {EPOCHS} epochs..."):
                        try:
                            metrics_history = train_model(merged_data)
                            st.session_state.metrics_history = metrics_history
                            st.session_state.models_initialized = True
                            
                            st.success("Training completed successfully!")
                            st.balloons()
                            
                            save_path = "trained_models"
                            os.makedirs(save_path, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Save models
                            torch.save(st.session_state.ts_generator.state_dict(), f"{save_path}/ts_generator_{timestamp}.pt")
                            torch.save(st.session_state.tab_generator.state_dict(), f"{save_path}/tab_generator_{timestamp}.pt")
                            torch.save(st.session_state.cross_modal_generator.state_dict(), f"{save_path}/cross_modal_{timestamp}.pt")
                            
                            metadata = {
                                'timestamp': timestamp,
                                'epochs': EPOCHS,
                                'features': FEATURES,
                                'tabular_features': TABULAR_FEATURES
                            }
                            with open(f"{save_path}/metadata_{timestamp}.json", 'w') as f:
                                json.dump(metadata, f)
                            
                            st.info(f"Models saved to {save_path} directory")
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                else:
                    st.error("Failed to load data. Please check your data files.")
    with tabs[3]:
        st.header("About This Project")
        
        st.markdown("""
        ## Multimodal Synthetic Healthcare Data Generation
        
        This application generates high-fidelity synthetic healthcare data that mirrors the complexity and 
        diversity of real-world medical records.
        
        ### Key Features:
        
        1. **Multimodal Data Synthesis**: Generates both tabular patient data and time-series measurements.
        2. **Cross-Modal Consistency**: Maintains realistic correlations between different data types.
        3. **Context-Aware Generation**: Tailors synthetic data to specific healthcare scenarios.
        4. **Validation Framework**: Comprehensive metrics to assess data quality and utility.
        
        ### Applications:
        
        - Supporting medical research
        - Developing and testing healthcare machine learning models
        - Sharing realistic healthcare data across institutions
        - Augmenting limited datasets for rare diseases or underrepresented populations
        
        ### How to Use:
        
        1. First, train the models using the Training tab
        2. Generate synthetic data with your desired parameters in the Data Generation tab
        3. Validate the quality of your data using the Validation tab
        """)
        
        st.subheader("System Architecture")
        
        mermaid_code = """
        graph TD
            A[Real Healthcare Data] --> B[Data Preprocessing]
            B --> D[Training Pipeline]
            D --> E[Time Series Generator]
            D --> F[Tabular Generator]
            D --> G[Cross-Modal Generator]
            E --> H[Synthetic Time Series Data]
            F --> I[Synthetic Tabular Data]
            G --> J[Cross-Modal Augmentation]
            H --> K[Validation & Metrics]
            I --> K
            K --> L[Data Export]
            K --> M[Visualization Dashboard]
        """
        
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

# ------------------- CLI: Train or Launch UI -------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model directly without launching UI')
    parser.add_argument('--generate', type=int, help='Generate specified number of synthetic records')
    parser.add_argument('--time-series-path', type=str, default="cleaned_merged_data.csv", help='Path to time-series data')
    parser.add_argument('--tabular-path', type=str, default=r"C:\Users\Asus\Downloads\syn\physionet_output\cleaned_tabular_data.csv", help='Path to tabular data')
    args = parser.parse_args()

    if args.train:
        print("Loading data...")
        merged_data, scalers, label_encoders = load_and_preprocess_data(args.time_series_path, args.tabular_path)
        if merged_data is not None:
            print(f"Training models for {EPOCHS} epochs...")
            globals()['scalers'] = scalers
            globals()['label_encoders'] = label_encoders
            try:
                metrics_history = train_model(merged_data)
                print("Training completed.")
            except Exception as e:
                print(f"Training failed: {str(e)}")
                raise
        else:
            print("Failed to load data. Please check your data files.")
    elif args.generate:
        if 'ts_generator' not in globals() or 'tab_generator' not in globals():
            print("Loading pretrained models...")
            model_dir = "trained_models"
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                model_timestamps = [f.split('_')[-1].split('.')[0] for f in files if f.startswith('metadata_')]
                if model_timestamps:
                    latest = max(model_timestamps)
                    print(f"Loading models from timestamp {latest}")
                    
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ts_generator = TimeSeriesGenerator().to(device)
                    tab_generator = TabularGenerator().to(device)
                    cross_modal_generator = CrossModalGenerator().to(device)
                    
                    ts_generator.load_state_dict(torch.load(f"{model_dir}/ts_generator_{latest}.pt"))
                    tab_generator.load_state_dict(torch.load(f"{model_dir}/tab_generator_{latest}.pt"))
                    cross_modal_generator.load_state_dict(torch.load(f"{model_dir}/cross_modal_{latest}.pt"))
                    
                    with open(f"{model_dir}/metadata_{latest}.json", 'r') as f:
                        metadata = json.load(f)
                    
                    print(f"Models loaded.")
                    
                    print(f"Generating {args.generate} synthetic records...")
                    synthetic_ts, synthetic_tab, conditions = generate_synthetic_data(args.generate)
                    
                    _, scalers, label_encoders = load_and_preprocess_data(args.time_series_path, args.tabular_path)
                    
                    ts_file, tab_file = save_synthetic_data(
                        synthetic_ts, synthetic_tab, conditions, 
                        scalers, label_encoders, args.generate
                    )
                    print(f"Data saved to {ts_file} and {tab_file}")
                else:
                    print("No pretrained models found. Please train models first.")
            else:
                print("No pretrained models found. Please train models first.")
        else:
            print(f"Generating {args.generate} synthetic records...")
            synthetic_ts, synthetic_tab, conditions = generate_synthetic_data(args.generate)
            
            ts_file, tab_file = save_synthetic_data(
                synthetic_ts, synthetic_tab, conditions, 
                globals().get('scalers', {}), globals().get('label_encoders', {}), 
                args.generate
            )
            print(f"Data saved to {ts_file} and {tab_file}")
    else:
        dashboard()
       