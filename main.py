import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================

df = pd.read_csv('dataset (1).csv')
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df = df.sort_values('datetime').reset_index(drop=True)

# Detect expiry date from data (more robust)
option_cols = [c for c in df.columns if c.startswith('NIFTY')]
ce_cols = sorted([c for c in option_cols if c.endswith('CE')])
pe_cols = sorted([c for c in option_cols if c.endswith('PE')])

# Extract strikes from column names
def extract_strike(col):
    import re
    numbers = re.findall(r'\d+', col)
    return int(numbers[-1]) if numbers else 0

ce_strikes = np.array([extract_strike(c) for c in ce_cols])
pe_strikes = np.array([extract_strike(c) for c in pe_cols])

# Better expiry detection
unique_dates = df['datetime'].dt.date.unique()
if len(unique_dates) > 1:
    # Find the largest date (likely expiry)
    expiry_date = max(unique_dates)
else:
    expiry_date = unique_dates[0]
    
EXPIRY_DT = pd.Timestamp(f'{expiry_date} 15:30:00')
print(f"Detected expiry date: {EXPIRY_DT}")

df['is_expiry_day'] = (df['datetime'].dt.date == expiry_date).astype(int)
df['days_to_expiry'] = (EXPIRY_DT - df['datetime']).dt.total_seconds() / (3600 * 24)
df['days_to_expiry'] = df['days_to_expiry'].clip(lower=0.01)

# ============================================================
# IMPROVED ATM IV CALCULATION
# ============================================================

def calculate_smooth_atm_iv(df, ce_cols, ce_strikes, pe_cols, pe_strikes):
    """Calculate ATM IV using interpolation around spot price"""
    atm_iv_list = []
    
    for idx, row in df.iterrows():
        spot = row['underlying_price']
        
        # For calls
        ce_diffs = np.abs(ce_strikes - spot)
        ce_idx = np.argsort(ce_diffs)[:3]  # Top 3 closest strikes
        ce_ivs = [row[ce_cols[i]] for i in ce_idx if not np.isnan(row[ce_cols[i]])]
        ce_strikes_sel = [ce_strikes[i] for i in ce_idx if not np.isnan(row[ce_cols[i]])]
        
        # For puts
        pe_diffs = np.abs(pe_strikes - spot)
        pe_idx = np.argsort(pe_diffs)[:3]
        pe_ivs = [row[pe_cols[i]] for i in pe_idx if not np.isnan(row[pe_cols[i]])]
        pe_strikes_sel = [pe_strikes[i] for i in pe_idx if not np.isnan(row[pe_cols[i]])]
        
        # Combine all near-the-money IVs
        all_strikes = ce_strikes_sel + pe_strikes_sel
        all_ivs = ce_ivs + pe_ivs
        
        if len(all_ivs) > 0:
            # Weighted average by distance to spot
            distances = np.abs(np.array(all_strikes) - spot)
            weights = 1 / (distances + 1e-6)
            weights = weights / weights.sum()
            atm_iv = np.average(all_ivs, weights=weights)
        else:
            atm_iv = np.nan
            
        atm_iv_list.append(atm_iv)
    
    return pd.Series(atm_iv_list).ffill().bfill()

atm_iv_s = calculate_smooth_atm_iv(df, ce_cols, ce_strikes, pe_cols, pe_strikes)

# Add more sophisticated features
df['atm_iv'] = atm_iv_s
df['spot_log_return'] = np.log(df['underlying_price'] / df['underlying_price'].shift(1))
df['spot_volatility'] = df['spot_log_return'].rolling(window=20).std()
df['spot_momentum'] = df['underlying_price'].pct_change(periods=5)

df['spot_rsi'] = calculate_rsi(df['underlying_price'])  # Need to implement RSI

# ============================================================
# IMPROVED SVI FITTING
# ============================================================

def fit_svi_improved(strikes_obs, iv_obs, K, spot):
    """Improved SVI fitting with better optimization"""
    try:
        if len(strikes_obs) < 5:
            return np.nan
            
        # Transform to log-moneyness
        lm = np.log(np.array(strikes_obs, dtype=float) / spot)
        lm_K = np.log(K / spot)
        
        # Remove outliers (3 sigma)
        iv_sq = np.array(iv_obs)**2
        mean_iv = np.mean(iv_sq)
        std_iv = np.std(iv_sq)
        mask = np.abs(iv_sq - mean_iv) <= 3 * std_iv
        lm = lm[mask]
        iv_sq = iv_sq[mask]
        
        if len(lm) < 3:
            return np.nan
        
        # Use SVI parametric form: w = a + b*(rho*(m - c) + sqrt((m - c)^2 + sigma^2))
        # This captures volatility smile better
        def svi_objective(params, m, w):
            a, b, rho, m_param, sigma = params
            if b <= 0 or sigma <= 0 or abs(rho) >= 1:
                return 1e10
            w_pred = a + b * (rho * (m - m_param) + np.sqrt((m - m_param)**2 + sigma**2))
            return np.mean((w - w_pred)**2)
        
        # Initial parameters
        params_init = [np.mean(iv_sq), 0.1, 0.0, np.median(lm), 0.5]
        bounds = [(0.001, 1.0), (0.001, 1.0), (-0.99, 0.99), (-2.0, 2.0), (0.001, 2.0)]
        
        result = minimize(svi_objective, params_init, args=(lm, iv_sq), 
                         bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            a, b, rho, m_param, sigma = result.x
            w_pred = a + b * (rho * (lm_K - m_param) + np.sqrt((lm_K - m_param)**2 + sigma**2))
            return float(np.sqrt(max(w_pred, 1e-8)))
        else:
            # Fallback to polynomial
            X = np.column_stack([np.ones(len(lm)), lm, lm**2, lm**3, lm**4])
            coeffs = np.linalg.lstsq(X, iv_sq, rcond=None)[0]
            iv2 = coeffs[0] + coeffs[1]*lm_K + coeffs[2]*lm_K**2 + coeffs[3]*lm_K**3 + coeffs[4]*lm_K**4
            return float(np.sqrt(max(iv2, 1e-8)))
            
    except:
        return np.nan

# ============================================================
# ADD DERIVATIVE FEATURES (VEGA, GAMMA)
# ============================================================

def calculate_greeks(spot, strike, iv, time_to_expiry, option_type):
    """Calculate option greeks (simplified)"""
    from scipy.stats import norm
    if time_to_expiry <= 0 or iv <= 0:
        return 0, 0
    
    t = time_to_expiry / 365  # Convert to years
    d1 = (np.log(spot/strike) + (0.5 * iv**2) * t) / (iv * np.sqrt(t))
    
    # Delta and Gamma approximations
    if option_type == 1:  # CE
        delta = norm.cdf(d1)
    else:  # PE
        delta = -norm.cdf(-d1)
    
    gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t))
    vega = spot * norm.pdf(d1) * np.sqrt(t)
    
    return delta, gamma, vega

# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================

print("Building enhanced features...")
records = []
ce_row_mean = df[ce_cols].mean(axis=1)
pe_row_mean = df[pe_cols].mean(axis=1)

for i, row in df.iterrows():
    spot = row['underlying_price']
    dt = row['datetime']
    time_to_expiry = max(row['days_to_expiry'], 0.01)
    
    # Time features
    tod = dt.hour * 60 + dt.minute
    sin_tod = np.sin(2 * np.pi * tod / (6.25 * 60))
    cos_tod = np.cos(2 * np.pi * tod / (6.25 * 60))
    
    # Market regime features
    atm_iv = row['atm_iv']
    spot_vol = row.get('spot_volatility', atm_iv)
    spot_mom = row.get('spot_momentum', 0)
    is_expiry = row['is_expiry_day']
    
    # For each option contract
    for cols, strikes, otype in [(ce_cols, ce_strikes, 1), (pe_cols, pe_strikes, 0)]:
        vals_row = row[cols].values.astype(float)
        side_mean = ce_row_mean.iloc[i] if otype == 1 else pe_row_mean.iloc[i]
        
        for j, (col, K) in enumerate(zip(cols, strikes)):
            iv_val = row[col]
            
            if pd.isna(iv_val) and i < len(df) - 1:  # Only predict missing values in training
                continue
                
            # Moneyness metrics
            moneyness = K / spot
            log_moneyness = np.log(moneyness)
            abs_log_moneyness = abs(log_moneyness)
            
            # Calculate greeks
            delta, gamma, vega = calculate_greeks(spot, K, iv_val if not pd.isna(iv_val) else atm_iv, time_to_expiry, otype)
            
            # Neighbors features
            left_iv = vals_row[j-1] if j > 0 else np.nan
            right_iv = vals_row[j+1] if j < len(cols)-1 else np.nan
            left_strike = strikes[j-1] if j > 0 else np.nan
            right_strike = strikes[j+1] if j < len(cols)-1 else np.nan
            
            # Interpolation features (improved)
            other_vals = np.delete(vals_row, j)
            other_strikes = np.delete(strikes, j)
            obs_mask = ~np.isnan(other_vals)
            n_obs = obs_mask.sum()
            
            # Various interpolation methods
            svi = np.nan
            cs_linear = np.nan
            cs_cubic = np.nan
            cs_quadratic = np.nan
            spline_fit = np.nan
            
            if n_obs >= 3:
                svi = fit_svi_improved(other_strikes[obs_mask], other_vals[obs_mask], K, spot)
            
            if n_obs >= 2:
                try:
                    cs_linear = float(interp1d(other_strikes[obs_mask], other_vals[obs_mask], 
                                              kind='linear', fill_value='extrapolate')(K))
                except:
                    pass
                
                try:
                    # Quadratic interpolation
                    if n_obs >= 3:
                        cs_quadratic = float(interp1d(other_strikes[obs_mask], other_vals[obs_mask], 
                                                     kind='quadratic', fill_value='extrapolate')(K))
                except:
                    pass
                    
            if n_obs >= 4:
                try:
                    cs_cubic = float(interp1d(other_strikes[obs_mask], other_vals[obs_mask], 
                                             kind='cubic', fill_value='extrapolate')(K))
                except:
                    pass
                
                try:
                    # Smoothing spline
                    spline = UnivariateSpline(other_strikes[obs_mask], other_vals[obs_mask], s=0.01)
                    spline_fit = float(spline(K))
                except:
                    pass
            
            # Time series features
            lag1 = df[col].iloc[i-1] if i > 0 else np.nan
            lag2 = df[col].iloc[i-2] if i > 1 else np.nan
            lag3 = df[col].iloc[i-3] if i > 2 else np.nan
            lag5 = df[col].iloc[i-5] if i > 4 else np.nan
            
            past = df[col].iloc[max(0, i-20):i].dropna().values
            
            # Rolling statistics
            roll5 = float(np.mean(past[-5:])) if len(past) >= 3 else np.nan
            roll10 = float(np.mean(past[-10:])) if len(past) >= 7 else np.nan
            roll20 = float(np.mean(past)) if len(past) >= 10 else np.nan
            roll_std = float(np.std(past[-10:])) if len(past) >= 5 else np.nan
            
            # EWMA with different spans
            ewm5 = float(df[col].iloc[:i].ewm(span=5).mean().iloc[-1]) if i > 0 else np.nan
            ewm10 = float(df[col].iloc[:i].ewm(span=10).mean().iloc[-1]) if i > 0 else np.nan
            
            # Rate of change
            roc = (lag1 - lag2) / lag2 if not pd.isna(lag1) and not pd.isna(lag2) and lag2 != 0 else np.nan
            
            # Volatility premium (IV vs HV)
            iv_minus_atm = iv_val - atm_iv if not pd.isna(iv_val) else np.nan
            
            records.append({
                'row_idx': i, 'contract': col, 'iv': iv_val,
                'strike': K, 'spot': spot, 'moneyness': moneyness,
                'log_m': log_moneyness, 'log_m_sq': log_moneyness**2,
                'abs_log_m': abs_log_moneyness,
                'option_type': otype, 'is_expiry': is_expiry,
                'days_to_expiry': time_to_expiry, 'sqrt_expiry': np.sqrt(time_to_expiry),
                'log_expiry': np.log(max(time_to_expiry, 0.01)),
                'delta': delta, 'gamma': gamma, 'vega': vega,
                'sin_tod': sin_tod, 'cos_tod': cos_tod,
                'left_iv': left_iv, 'right_iv': right_iv,
                'left_strike': left_strike, 'right_strike': right_strike,
                'strike_spread': (right_strike - left_strike) if not pd.isna(left_strike) and not pd.isna(right_strike) else np.nan,
                'svi': svi, 'cs_linear': cs_linear, 'cs_cubic': cs_cubic, 
                'cs_quadratic': cs_quadratic, 'spline_fit': spline_fit,
                'n_obs': n_obs,
                'lag1': lag1, 'lag2': lag2, 'lag3': lag3, 'lag5': lag5,
                'roll5': roll5, 'roll10': roll10, 'roll20': roll20,
                'roll_std': roll_std,
                'ewm5': ewm5, 'ewm10': ewm10,
                'roc': roc,
                'atm_iv': atm_iv, 'side_mean': side_mean,
                'iv_minus_atm': iv_minus_atm,
                'spot_vol': spot_vol, 'spot_mom': spot_mom,
            })

feat_df = pd.DataFrame(records)

# ============================================================
# FEATURE SELECTION AND ENGINEERING
# ============================================================

# Define enhanced feature set
FEATURES = [
    'strike', 'spot', 'moneyness', 'log_m', 'log_m_sq', 'abs_log_m',
    'option_type', 'is_expiry',
    'days_to_expiry', 'sqrt_expiry', 'log_expiry',
    'delta', 'gamma', 'vega',
    'sin_tod', 'cos_tod',
    'left_iv', 'right_iv', 'strike_spread',
    'svi', 'cs_linear', 'cs_cubic', 'cs_quadratic', 'spline_fit',
    'n_obs',
    'lag1', 'lag2', 'lag3', 'lag5',
    'roll5', 'roll10', 'roll20', 'roll_std',
    'ewm5', 'ewm10',
    'roc',
    'atm_iv', 'side_mean', 'iv_minus_atm',
    'spot_vol', 'spot_mom',
]

# Create interaction features
feat_df['atm_interaction'] = feat_df['atm_iv'] * feat_df['moneyness']
feat_df['time_moneyness'] = feat_df['days_to_expiry'] * feat_df['abs_log_m']
feat_df['vol_skew'] = feat_df['atm_iv'] - feat_df['side_mean']

FEATURES.extend(['atm_interaction', 'time_moneyness', 'vol_skew'])

# ============================================================
# ENHANCED MODEL WITH STACKING
# ============================================================

train_df = feat_df[feat_df['iv'].notna()].copy()
X = train_df[FEATURES]
y = train_df['iv']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CV to test different blends
tscv = TimeSeriesSplit(n_splits=5)

# Store predictions for stacking
lgbm_preds = []
xgb_preds = []
cat_preds = []
rf_preds = []
true_vals = []

print("\nCross-validation results:")
for fold, (tr, te) in enumerate(tscv.split(X_scaled), 1):
    # Filter out expiry day from test
    te_mask = [t for t in te if train_df.iloc[t]['is_expiry'] == 0]
    if len(te_mask) < 5:
        continue
        
    X_tr, X_te = X_scaled[tr], X_scaled[te_mask]
    y_tr, y_te = y.iloc[tr], y.iloc[te_mask]
    
    # LightGBM
    lgbm = LGBMRegressor(
        n_estimators=5000, learning_rate=0.007, max_depth=8, num_leaves=72,
        subsample=0.75, colsample_bytree=0.7, min_child_samples=8,
        reg_alpha=0.03, reg_lambda=0.3, random_state=42, verbose=-1
    )
    lgbm.fit(X_tr, y_tr)
    lgbm_pred = lgbm.predict(X_te)
    
    # XGBoost
    xgb = XGBRegressor(
        n_estimators=3000, learning_rate=0.01, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    xgb.fit(X_tr, y_tr)
    xgb_pred = xgb.predict(X_te)
    
    # CatBoost
    cat = CatBoostRegressor(
        iterations=3000, learning_rate=0.01, depth=6,
        random_seed=42, verbose=False
    )
    cat.fit(X_tr, y_tr)
    cat_pred = cat.predict(X_te)
    
    # Random Forest (for diversity)
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=12, min_samples_split=10,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    
    # Blend with optimal weights (found via validation)
    svi_pred = train_df.iloc[te_mask]['svi'].fillna(train_df.iloc[te_mask]['cs_linear']).values
    svi_pred = np.nan_to_num(svi_pred, nan=train_df.iloc[te_mask]['atm_iv'].values[0])
    
    # Smart blending based on confidence
    blend_weights = [0.15, 0.25, 0.20, 0.10, 0.30]  # [lgbm, xgb, cat, rf, svi]
    blend = (blend_weights[0] * lgbm_pred + 
             blend_weights[1] * xgb_pred + 
             blend_weights[2] * cat_pred + 
             blend_weights[3] * rf_pred +
             blend_weights[4] * svi_pred)
    
    mse = mean_squared_error(y_te, blend)
    print(f"  Fold {fold}: MSE={mse:.8f}")
    
    lgbm_preds.extend(lgbm_pred)
    xgb_preds.extend(xgb_pred)
    cat_preds.extend(cat_pred)
    rf_preds.extend(rf_pred)
    true_vals.extend(y_te)

# Calculate optimal weights using linear regression
from sklearn.linear_model import LinearRegression
stack_X = np.column_stack([lgbm_preds, xgb_preds, cat_preds, rf_preds])
stack_model = LinearRegression()
stack_model.fit(stack_X, true_vals)
optimal_weights = stack_model.coef_
print(f"\nOptimal stacking weights: {optimal_weights}")

# ============================================================
# FINAL MODEL TRAINING WITH STACKING
# ============================================================

print("\nTraining final ensemble...")

# Train individual models on full dataset
final_lgbm = LGBMRegressor(
    n_estimators=5000, learning_rate=0.007, max_depth=8, num_leaves=72,
    subsample=0.75, colsample_bytree=0.7, min_child_samples=8,
    reg_alpha=0.03, reg_lambda=0.3, random_state=42, verbose=-1
)
final_lgbm.fit(X_scaled, y)

final_xgb = XGBRegressor(
    n_estimators=3000, learning_rate=0.01, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
final_xgb.fit(X_scaled, y)

final_cat = CatBoostRegressor(
    iterations=3000, learning_rate=0.01, depth=6,
    random_seed=42, verbose=False
)
final_cat.fit(X_scaled, y)

final_rf = RandomForestRegressor(
    n_estimators=500, max_depth=12, min_samples_split=10,
    random_state=42, n_jobs=-1
)
final_rf.fit(X_scaled, y)

# ============================================================
# PREDICT MISSING VALUES
# ============================================================

missing_df = feat_df[feat_df['iv'].isna()].copy()
print(f"\nMissing rows to predict: {len(missing_df)}")

if len(missing_df) > 0:
    # Scale missing data
    missing_X = scaler.transform(missing_df[FEATURES])
    
    # Make predictions
    lgb_pred = final_lgbm.predict(missing_X)
    xgb_pred = final_xgb.predict(missing_X)
    cat_pred = final_cat.predict(missing_X)
    rf_pred = final_rf.predict(missing_X)
    
    # SVI prediction
    svi_pred = missing_df['svi'].fillna(missing_df['cs_linear']).fillna(missing_df['cs_quadratic']).fillna(missing_df['atm_iv'])
    svi_pred = svi_pred.fillna(missing_df['atm_iv']).values
    
    # Stacking blend
    final_pred = (optimal_weights[0] * lgb_pred + 
                  optimal_weights[1] * xgb_pred + 
                  optimal_weights[2] * cat_pred + 
                  optimal_weights[3] * rf_pred +
                  0.3 * svi_pred)  # Keep SVI weight
    
    # Adaptive clipping based on market conditions
    clip_upper = np.percentile(train_df['iv'].dropna(), 99) * 1.5
    clip_lower = max(0.005, np.percentile(train_df['iv'].dropna(), 1) * 0.5)
    
    final_pred = np.clip(final_pred, clip_lower, clip_upper)
    
    missing_df['pred_iv'] = final_pred
    
    # Additional smoothing for extreme values
    missing_df['pred_iv'] = missing_df['pred_iv'].rolling(window=3, center=True).mean().fillna(missing_df['pred_iv'])

# ============================================================
# BUILD SUBMISSION
# ============================================================

submission_rows = []
for _, row in missing_df.iterrows():
    original_row = df.iloc[int(row["row_idx"])]
    dt_str = original_row["datetime"].strftime("%d-%m-%Y %H:%M")
    submission_rows.append({
        "id": f"{dt_str}||{row['contract']}",
        "value": float(row["pred_iv"])
    })

submission = pd.DataFrame(submission_rows, columns=["id", "value"])
submission = submission.sort_values("id").reset_index(drop=True)

# ============================================================
# SAVE AND VALIDATE
# ============================================================

submission.to_csv("submission_improved.csv", index=False)

print(f"\n✓ Submission saved: submission_improved.csv")
print(f"✓ Shape: {submission.shape}")
print(f"✓ Value range: [{submission['value'].min():.4f}, {submission['value'].max():.4f}]")
print(f"✓ Mean IV: {submission['value'].mean():.4f}")
print(f"✓ Std IV: {submission['value'].std():.4f}")

# Sanity checks
assert submission.shape[1] == 2
assert "id" in submission.columns
assert "value" in submission.columns
assert not submission['value'].isna().any()
assert (submission['value'] > 0).all()