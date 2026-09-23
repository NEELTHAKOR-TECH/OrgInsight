"""Inspect ML appropriateness and data structure before modifying anything."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/processed/organizations_clean.csv')

print('=== DATASET SHAPE ===')
print(df.shape)
print(df.dtypes)

print()
print('=== TARGET: size_band ===')
sb = df['size_band'].value_counts().sort_index()
for k, v in sb.items():
    print(f'  {k:12s}: {v:>7,}  ({v/len(df)*100:.2f}%)')
print(f'  CLASS IMBALANCE RATIO (max/min): {sb.max()/sb.min():.1f}x')

print()
print('=== TARGET: Number of employees ===')
emp = df['Number of employees']
print(f'  min={emp.min()}  max={emp.max()}  mean={emp.mean():.1f}  std={emp.std():.1f}')
print(f'  skewness={emp.skew():.4f}  kurtosis={emp.kurtosis():.4f}')
print(f'  uniform? Std/Mean = {emp.std()/emp.mean():.3f}')

print()
print('=== CORRELATION MATRIX: features vs target ===')
le_i = LabelEncoder()
le_c = LabelEncoder()
le_s = LabelEncoder()
df['ind_enc'] = le_i.fit_transform(df['Industry'])
df['cty_enc'] = le_c.fit_transform(df['Country'])
df['sec_enc'] = le_s.fit_transform(df['broad_sector'])
size_map = {'Micro':0,'Small':1,'Medium':2,'Large':3,'Enterprise':4}
df['target_num'] = df['size_band'].map(size_map)

for feat in ['company_age','Founded','ind_enc','cty_enc','sec_enc']:
    r = df[feat].corr(df['target_num'])
    print(f'  {feat:20s} vs target: r={r:.6f}')

print()
print('=== SPEARMAN CORRELATION vs target (size_band) ===')
from scipy.stats import spearmanr
for feat in ['company_age','Founded','ind_enc','cty_enc','sec_enc']:
    rho, p = spearmanr(df[feat], df['target_num'])
    print(f'  {feat:20s}: rho={rho:.6f}  p={p:.4f}')

print()
print('=== BASELINE (dummy): most frequent class = Enterprise ===')
from sklearn.dummy import DummyClassifier
X_dummy = df[['company_age']].values
y = df['target_num'].values
from sklearn.model_selection import train_test_split
Xt, Xv, yt, yv = train_test_split(X_dummy, y, test_size=0.2, random_state=42, stratify=y)
dum = DummyClassifier(strategy='most_frequent')
dum.fit(Xt, yt)
dummy_acc = dum.score(Xv, yv)
print(f'  Most-frequent-class accuracy = {dummy_acc:.4f} ({dummy_acc*100:.2f}%)')
dum2 = DummyClassifier(strategy='stratified', random_state=42)
dum2.fit(Xt, yt)
dummy_acc2 = dum2.score(Xv, yv)
print(f'  Stratified random accuracy   = {dummy_acc2:.4f} ({dummy_acc2*100:.2f}%)')

print()
print('=== REGRESSION TARGET: Number of employees directly ===')
print('  Direct regression would predict exact employee count (1-9999)')
print('  Uniform distribution makes this a hard regression problem')
print('  R2 of a mean predictor = 0.0 by definition')
# What would random forest regression get?
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
X_reg = df[['company_age','Founded','ind_enc','cty_enc','sec_enc']].values
y_reg = df['Number of employees'].values
Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
# Quick test with small forest
rfr = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rfr.fit(Xtr, ytr)
pred = rfr.predict(Xte)
mae = mean_absolute_error(yte, pred)
rmse = np.sqrt(mean_squared_error(yte, pred))
r2 = r2_score(yte, pred)
print(f'  RF Regressor (n=50, d=10): MAE={mae:.1f}  RMSE={rmse:.1f}  R2={r2:.4f}')
# Null model RMSE
null_rmse = np.sqrt(mean_squared_error(yte, np.full_like(yte, float(ytr.mean()))))
print(f'  Null (mean) model: RMSE={null_rmse:.1f}')

print()
print('=== LEAKAGE CHECK: is size_band derived ONLY from employees? ===')
size_map_check = {'Micro':(0,49),'Small':(50,249),'Medium':(250,999),
                  'Large':(1000,4999),'Enterprise':(5000,99999)}
for band, (lo, hi) in size_map_check.items():
    sub = df[df['size_band']==band]
    outside = sub[(sub['Number of employees']<lo) | (sub['Number of employees']>hi)]
    print(f'  {band}: {len(sub)} orgs, {len(outside)} outside expected range')
