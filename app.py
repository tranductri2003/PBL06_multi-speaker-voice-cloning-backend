from flask import Flask

app = Flask(__name__)

@app.route('/')
def welcome():
    return '''
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chào mừng!</title>
        <style>
            body {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(to right, #ff7e5f, #feb47b);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-align: center;
            }
            .container {
                background: rgba(0, 0, 0, 0.5);
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            }
            h1 {
                font-size: 3em;
                margin-bottom: 20px;
            }
            p {
                font-size: 1.2em;
                margin-bottom: 30px;
            }
            img {
                max-width: 300px;
                margin-top: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .author {
                margin-top: 30px;
                font-size: 1em;
                color: #ffd700;
            }
            .icon {
                font-size: 2em;
                margin-top: 20px;
            }
            @media (max-width: 600px) {
                h1 {
                    font-size: 2em;
                }
                p {
                    font-size: 1em;
                }
                img {
                    max-width: 200px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chào mừng đến với trang của tôi!</h1>
            <p>Trang web này được xây dựng bằng Flask, nhằm mục đích chia sẻ kiến thức và kinh nghiệm về lập trình.</p>
            <div class="icon">😊</div>
            <div class="icon">😊</div>
            <div class="icon">😊</div>
            <div class="icon">😊</div>
            <div class="icon">😊</div>

            <div class="author">
                <p>Được thiết kế bởi: <strong>Tâm Tuấn Phát Trí</strong></p>
                <p>Author của web</p>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
