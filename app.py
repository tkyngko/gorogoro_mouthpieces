# 必要なモジュールのインポート
import torch
from gorogoro import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
import os
import cv2
from seg import remove_background  # seg.pyから背景除去関数をインポート

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # 学習済みモデルの重み（gorogoro_weights.pth）を読み込み
    net.load_state_dict(torch.load(r"C:\Users\reine\OneDrive\ドキュメント\gorogoro_flask_5\src\gorogoro_weights_resnet18.pth", map_location=torch.device('cpu')))
    
    # データの前処理
    img = transform(img)
    img = img.unsqueeze(0)  # 1次元増やす
    # 推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

# 推論したラベルから名前を返す関数
def getName(label):
    print("Label received:", label)  # 受け取ったラベルを確認
    names = [
        'Beechler', 'Claude Lakey', 'MC Gregory', 'Morgan',
        'Otto Link_Tone edge_Slant', 'Otto Link_Tone edge',
        'Otto Link_Tone edge_Vintage Model', 'Wood Stone_Model 46',
        'Wood Stone_Traditional Jazz', 'else'
    ]
    return names[label[0]]  # 配列の最初の要素を返す

# Flask のインスタンスを作成
app = Flask(__name__)

# 静的フォルダの作成
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/segmented', exist_ok=True)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = {'png', 'jpg', 'gif', 'jpeg'}

# 拡張子が適切かどうかをチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods=['GET', 'POST'])
def predicts():
    # リクエストがPOSTかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'image' not in request.files:
            return redirect(request.url)

        # データの取り出し
        file = request.files['image']
        # ファイルのチェック
        if file and allowed_file(file.filename):
            # アップロードされた画像を保存
            img_path = os.path.join('static/uploads', file.filename)
            file.save(img_path)

            # 背景除去処理
            img_background_removed = remove_background(img_path)

            if img_background_removed is not None:
                # 背景除去された画像をPIL形式に変換
                img_pil = Image.fromarray(img_background_removed)

                # 推論を行う
                pred = predict(img_pil)
                gorogoroName_ = getName(pred)

                # 画像のバッファを作成してbase64エンコード（元の画像）
                buf_original = io.BytesIO()
                original_image = Image.open(img_path)  # 元の画像を読み込む
                original_image.save(buf_original, 'PNG')
                original_base64_str = base64.b64encode(buf_original.getvalue()).decode('utf-8')
                original_base64_data = f'data:image/png;base64,{original_base64_str}'

                # 画像のバッファを作成してbase64エンコード（背景処理された画像）
                buf = io.BytesIO()
                img_pil.save(buf, 'PNG')
                base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                base64_data = f'data:image/png;base64,{base64_str}'

                # 結果を表示するテンプレートに渡す
                return render_template('result.html', gorogoroName=gorogoroName_, 
                                       original_image=original_base64_data, 
                                       segmented_image=base64_data)

    # GETメソッドの定義
    return render_template('index.html')

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
