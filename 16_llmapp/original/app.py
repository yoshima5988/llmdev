# VS Codeのデバッグ実行で `from chatbot.graph` でエラーを出さない対策
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, make_response 
from original.message import get_bot_response, memory

# Flaskアプリケーションのセットアップ
app = Flask(__name__)

system_message = ''

@app.route('/', methods=['GET', 'POST'])
def index():

    # GETリクエスト時は初期メッセージ表示
    if request.method == 'GET':
        # メモリをクリア
        memory.storage.clear()
        # 対話履歴を初期化
        response = make_response(render_template('index.html', messages=[]))
        return response

    # ユーザーからのメッセージを取得
    user_message = request.form['user_message']
    system_message = request.form['system_message']

    # ボットのレスポンスを取得（メモリに保持）
    messages = get_bot_response(user_message, system_message, memory)

    # レスポンスを返す
    return make_response(render_template('index.html', messages=messages))

if __name__ == '__main__':
    app.run(debug=True)