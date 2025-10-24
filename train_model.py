# train_model.py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os


SALES_CSV = "sales.csv"
MODELS_DIR = "."


# Carregar vendas
sales = pd.read_csv(SALES_CSV, parse_dates=["date"]) # date,product_id,units


# Preprocess
sales = sales.sort_values("date")
# Agregar por dia e produto (garante linhas contínuas)
daily = sales.groupby(["product_id", "date"]).sum().reset_index()


# Criar série diária completa por produto (preenche zeros nos dias sem venda)
products = daily['product_id'].unique()


models_info = {}


for sku in products:
df = daily[daily['product_id'] == sku].set_index('date').asfreq('D').fillna(0)
df['product_id'] = sku
df['day'] = df.index.day
df['month'] = df.index.month
df['weekday'] = df.index.weekday
# Lag features
df['lag1'] = df['units'].shift(1).fillna(0)
df['lag2'] = df['units'].shift(2).fillna(0)
df['lag7'] = df['units'].shift(7).fillna(0)
df = df.reset_index()


# Remover primeiras linhas com NaNs (já foram preenchidas com 0)
features = ['day', 'month', 'weekday', 'lag1', 'lag2', 'lag7']
X = df[features]
y = df['units']


# Se poucos dados, fica com média móvel (não treina modelo)
if len(df) < 30:
print(f"SKU {sku} tem dados reduzidos ({len(df)} linhas) — vai usar fallback média móvel.")
models_info[sku] = {'model': None, 'mae': None}
continue


# Train/test split por série temporal: usar as últimas 20% como teste
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"SKU {sku} treinado — MAE: {mae:.2f}")


model_path = os.path.join(MODELS_DIR, f"model_{sku}.pkl")
joblib.dump(model, model_path)
models_info[sku] = {'model': model_path, 'mae': mae}


# Guardar sumário de modelos
import json
with open('models_summary.json', 'w') as f:
json.dump(models_info, f, indent=2)


print("Treino concluído. Modelos guardados.")