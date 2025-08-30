from flask import Flask, request, jsonify
import pandas as pd
import traceback
import joblib
import numpy as np

app = Flask(__name__)

numeric_cols = ['kepadatan_penduduk', 'tingkat_pendidikan',
                'tingkat_pengangguran', 'akses_air_bersih', 'akses_listrik',
                'fasilitas_kesehatan', 'jalan_aspal',
                'luas_sawah', 'pendapatan_perkapita']
cat_cols = ['jenis_wilayah']
preprocess = joblib.load("preprocessor.pkl")
cnn = joblib.load("BPS.pkl")


@app.route('/classify_uber', methods=['POST'])
def uber_predict():
    try:
        raw_json = request.json
        data_list = None

        # --- Extract data from JSON ---
        if isinstance(raw_json, list) and len(raw_json) > 0 and isinstance(raw_json[0], dict) and "data" in raw_json[0]:
            data_list = raw_json[0]["data"]
        elif isinstance(raw_json, dict) and "data" in raw_json:
            data_list = raw_json["data"]
        elif isinstance(raw_json, list):
            data_list = raw_json

        if data_list is None:
            return jsonify({"error": "Format JSON input tidak dikenali atau kosong."}), 400
        n_samples = 5000
        data = {
            'provinsi_id': np.random.randint(1, 35, n_samples),  # 34 provinsi
            # ~514 kab/kota
            'kabupaten_id': np.random.randint(1, 515, n_samples),
            'kepadatan_penduduk': np.random.lognormal(5, 1.5, n_samples),
            # rata2 tahun sekolah
            'tingkat_pendidikan': np.random.normal(8.5, 2.5, n_samples),
            'tingkat_pengangguran': np.random.gamma(2, 3, n_samples),  # %
            'akses_air_bersih': np.random.beta(7, 2, n_samples) * 100,  # %
            'akses_listrik': np.random.beta(8, 1.5, n_samples) * 100,  # %
            # per 10k penduduk
            'fasilitas_kesehatan': np.random.poisson(15, n_samples),
            'jalan_aspal': np.random.beta(5, 3, n_samples) * 100,  # %
            'luas_sawah': np.random.exponential(5000, n_samples),  # hektar
            # rupiah/bulan
            'pendapatan_perkapita': np.random.lognormal(14.5, 0.8, n_samples),
            'jenis_wilayah': np.random.choice(['urban', 'rural'], n_samples, p=[0.6, 0.4])
        }

        # Buat DataFrame dari input
        df = pd.DataFrame(data)

        # kalau masih ada kolom 'data', pecah lagi
        if 'data' in df.columns:
            df = pd.json_normalize(df['data'])

        # Replace string null jadi NaN
        df.replace(["null", "NULL", "NaN"], np.nan, inplace=True)
        X = df[numeric_cols + cat_cols]
        X_processed = preprocess.transform(X)
        y_pred = cnn.predict(X_processed)
        proba = cnn.predict_proba(X_processed)
        df['prob'] = proba[:, 1]
        df['Prediction'] = y_pred
        return jsonify(df.to_dict(orient="records")), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500


@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "OK"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
