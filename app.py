from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================
# CONFIGURACIÓN — rellenar cuando esté lista la BD
MONGO_URI        = "dZPWN7tTOvb7L5fr@hack4her.4wagxyf.mongodb.net/ProcessedData?appName=Hack4Her"
MONGO_DB         = "ProcessedData"
MONGO_COLLECTION = "cliens"
# ============================================

# Cargar modelo
modelo = joblib.load('modelo_churn.pkl')

# Conectar a MongoDB
cliente_mongo = MongoClient(MONGO_URI)
db            = cliente_mongo[MONGO_DB]
col_clientes  = db[MONGO_COLLECTION]

# Columnas requeridas
COLUMNAS_REQUERIDAS = ['customer_id', 'calmonth', 'num_transacciones', 'uni_boxes_sold_m']
COLUMNAS_OPCIONALES = ['num_coolers', 'num_doors']

def get_info_clientes(customer_ids):
    """Consulta MongoDB para obtener info estática de los clientes"""
    try:
        docs = col_clientes.find(
            {'customer_id': {'$in': customer_ids}},
            {'_id': 0, 'customer_id': 1, 'territory_d': 1,
             'comercial_subchannel_d': 1, 'rtm_customer_size_d': 1}
        )
        return pd.DataFrame(list(docs))
    except Exception as e:
        print(f"Error consultando MongoDB: {e}")
        return pd.DataFrame()

def validar_y_limpiar(df):
    rechazados = []

    # 1. Verificar columnas requeridas
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        return None, None, f"Columnas faltantes: {faltantes}"

    # 2. Rellenar opcionales con 0
    for col in COLUMNAS_OPCIONALES:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # 3. Validar filas
    filas_validas = []
    for idx, row in df.iterrows():
        razon = None

        if pd.isna(row['customer_id']) or str(row['customer_id']).strip() == '':
            razon = 'customer_id vacío'
        elif pd.isna(row['calmonth']):
            razon = 'calmonth vacío'
        elif not str(int(row['calmonth'])).startswith(('20', '19')) or len(str(int(row['calmonth']))) != 6:
            razon = 'calmonth formato incorrecto — debe ser YYYYMM'
        elif pd.isna(row['num_transacciones']):
            razon = 'num_transacciones vacío'
        elif pd.isna(row['uni_boxes_sold_m']):
            razon = 'uni_boxes_sold_m vacío'
        elif row['num_transacciones'] < 0 or row['uni_boxes_sold_m'] < 0:
            razon = 'valores negativos no permitidos'

        if razon:
            rechazados.append({
                'fila'       : int(idx) + 1,
                'customer_id': str(row['customer_id']) if not pd.isna(row['customer_id']) else None,
                'razon'      : razon
            })
        else:
            filas_validas.append(idx)

    df_limpio = df.loc[filas_validas].copy()
    return df_limpio, rechazados, None

def calcular_features(df):
    df = df.sort_values(['customer_id', 'calmonth'])

    # Excluir último mes para evitar leakage
    df_sin_ultimo = df.groupby('customer_id').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    # Clientes con menos de 3 meses
    conteo                = df.groupby('customer_id').size()
    clientes_sin_historial = conteo[conteo < 3].index.tolist()

    # Agregar por cliente
    agg = df.groupby('customer_id').agg(
        meses_activo      = ('calmonth', 'count'),
        avg_transacciones = ('num_transacciones', 'mean'),
        avg_cajas         = ('uni_boxes_sold_m', 'mean'),
        avg_coolers       = ('num_coolers', 'mean'),
        avg_doors         = ('num_doors', 'mean'),
    ).reset_index()

    # Features temporales
    temp = df_sin_ultimo.groupby('customer_id').agg(
        ventas_ultimo_mes    = ('uni_boxes_sold_m', 'last'),
        ventas_penultimo     = ('uni_boxes_sold_m', lambda x: x.iloc[-2] if len(x) >= 2 else x.iloc[-1]),
        ventas_antepenultimo = ('uni_boxes_sold_m', lambda x: x.iloc[-3] if len(x) >= 3 else x.iloc[-1]),
        meses_sin_venta      = ('num_transacciones', lambda x: (x == 0).sum()),
        max_coolers          = ('num_coolers', 'max'),
        ultimo_cooler        = ('num_coolers', 'last'),
    ).reset_index()

    temp['perdio_cooler'] = ((temp['max_coolers'] > 0) & (temp['ultimo_cooler'] == 0)).astype(int)

    df_final = agg.merge(temp, on='customer_id', how='left')

    # Consultar info de clientes en MongoDB
    ids           = df_final['customer_id'].tolist()
    info_clientes = get_info_clientes(ids)

    if not info_clientes.empty:
        df_final = df_final.merge(info_clientes, on='customer_id', how='left')
    else:
        # Si MongoDB no responde, usar valores default
        df_final['territory_d']             = 'Desconocido'
        df_final['comercial_subchannel_d']  = 'Desconocido'
        df_final['rtm_customer_size_d']     = 'Desconocido'

    # Rellenar nulos de categoricas (clientes no encontrados en MongoDB)
    for col in ['territory_d', 'comercial_subchannel_d', 'rtm_customer_size_d']:
        df_final[col] = df_final[col].fillna('Desconocido')

    return df_final, clientes_sin_historial

def generar_razones_y_propuestas(row):
    razones   = []
    propuestas = []

    if row['ventas_ultimo_mes'] < row['ventas_penultimo'] < row['ventas_antepenultimo']:
        razones.append("Ventas en caída los últimos 3 meses")
        propuestas.append("Ofrecer promoción o descuento especial")

    if row['meses_sin_venta'] >= 2:
        razones.append(f"Sin actividad en {int(row['meses_sin_venta'])} meses")
        propuestas.append("Contactar urgente para entender situación")

    if row['perdio_cooler'] == 1:
        razones.append("Perdió enfriador asignado")
        propuestas.append("Revisar y restablecer cobertura de enfriador")

    if row.get('rtm_customer_size_d') == 'Mini' and row['prob_churn'] > 0.6:
        razones.append("Cliente pequeño con alta probabilidad de abandono")
        propuestas.append("Asignar plan de retención para clientes pequeños")

    if row['meses_activo'] <= 3:
        razones.append("Cliente nuevo, aún no consolidado")
        propuestas.append("Reforzar acompañamiento en primeros meses")

    if row['ventas_ultimo_mes'] == 0:
        razones.append("Sin ventas en el último mes registrado")
        propuestas.append("Llamada preventiva para consultar situación")

    return (
        " | ".join(razones)    if razones    else "Sin factores críticos detectados",
        " | ".join(propuestas) if propuestas else "Mantener seguimiento regular"
    )

@app.route('/predecir', methods=['POST'])
def predecir():
    # 1. Verificar archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400

    archivo = request.files['file']
    if not archivo.filename.endswith('.csv'):
        return jsonify({'error': 'Solo se aceptan archivos CSV'}), 400

    # 2. Leer CSV
    try:
        df = pd.read_csv(archivo)
    except Exception as e:
        return jsonify({'error': f'Error al leer el CSV: {str(e)}'}), 400

    # 3. Validar y limpiar
    df_limpio, rechazados, error = validar_y_limpiar(df)
    if error:
        return jsonify({'error': error}), 400

    if len(df_limpio) == 0:
        return jsonify({
            'procesados' : [],
            'rechazados' : rechazados,
            'mensaje'    : 'Ninguna fila pasó la validación'
        }), 200

    # 4. Calcular features
    df_features, sin_historial = calcular_features(df_limpio)

    # 5. Separar clientes sin historial suficiente
    df_predecir       = df_features[~df_features['customer_id'].isin(sin_historial)].copy()
    sin_historial_resp = [
        {'customer_id': cid, 'razon': 'Menos de 3 meses de historial — predicción no disponible'}
        for cid in sin_historial
    ]

    if len(df_predecir) == 0:
        return jsonify({
            'procesados'    : [],
            'sin_historial' : sin_historial_resp,
            'rechazados'    : rechazados,
            'mensaje'       : 'Todos los clientes tienen historial insuficiente'
        }), 200

    # 6. Codificar categóricas
    le = LabelEncoder()
    for col, enc_col in [('territory_d', 'territorio_enc'),
                          ('comercial_subchannel_d', 'subchannel_enc'),
                          ('rtm_customer_size_d', 'tamaño_enc')]:
        df_predecir[enc_col] = le.fit_transform(df_predecir[col].astype(str))

    # 7. Correr modelo
    X = df_predecir[['territorio_enc', 'subchannel_enc', 'tamaño_enc',
                      'meses_activo', 'avg_transacciones', 'avg_cajas',
                      'avg_coolers', 'avg_doors', 'ventas_ultimo_mes',
                      'ventas_penultimo', 'ventas_antepenultimo',
                      'meses_sin_venta', 'perdio_cooler']]

    probs = modelo.predict_proba(X)[:, 1]

    # 8. Armar respuesta
    procesados = []
    for i, (_, row) in enumerate(df_predecir.iterrows()):
        prob = float(probs[i])
        if prob <= 0.3:
            nivel = 'Bajo'
        elif prob <= 0.6:
            nivel = 'Medio'
        else:
            nivel = 'Alto'

        row['prob_churn'] = prob
        razones, propuestas = generar_razones_y_propuestas(row)

        procesados.append({
            'customer_id' : row['customer_id'],
            'prob_churn'  : round(prob, 4),
            'nivel_riesgo': nivel,
            'razones'     : razones,
            'propuestas'  : propuestas
        })

    return jsonify({
        'procesados'      : procesados,
        'sin_historial'   : sin_historial_resp,
        'rechazados'      : rechazados,
        'total_procesados': len(procesados),
        'total_rechazados': len(rechazados) + len(sin_historial_resp)
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)