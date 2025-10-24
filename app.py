# app.py
pred = model.predict(feat)[0]
pred = max(0, pred)
preds.append(pred)
# atualizar lags
lag2 = lag1
lag1 = pred
return float(np.sum(preds))
# fallback: média móvel
recent = sales['units'].tail(14)
mean14 = float(recent.mean()) if len(recent) > 0 else 0.0
return mean14 * days




class RestockRequest(BaseModel):
sku: str
lead_time_days: Optional[int] = 7
safety_stock: Optional[int] = 5




@app.get('/products')
def list_products():
return products_df.to_dict(orient='records')




@app.post('/predict_restock')
def predict_restock(req: RestockRequest):
sku = req.sku
lead = req.lead_time_days
safety = req.safety_stock


# current stock
cur = stock_df[stock_df['product_id'] == sku]
if cur.empty:
raise HTTPException(status_code=404, detail='SKU não encontrado no stock')
current_stock = int(cur['current_stock'].values[0])


predicted = predict_future_sales_with_model(sku, days=lead)


# reorder point from products file (fallback 0)
rp_row = products_df[products_df['product_id'] == sku]
reorder_point = int(rp_row['reorder_point'].values[0]) if not rp_row.empty else 0


should_reorder = (current_stock <= reorder_point) or (predicted + safety > current_stock)
suggested_qty = math.ceil(max(0, (predicted + safety) - current_stock))


return {
'sku': sku,
'current_stock': current_stock,
'predicted_demand_next_{}_days'.format(lead): round(predicted, 2),
'safety_stock': safety,
'should_reorder': bool(should_reorder),
'suggested_quantity': int(suggested_qty)
}




@app.post('/create_order')
def create_order(order: dict):
# Simula criar uma ordem — pode ser expandido para guardar em DB
return {'status': 'created', 'order': order}




@app.get('/')
def root():
return {'message': 'Assistente Reposição API — acede a /docs para ver os endpoints'}