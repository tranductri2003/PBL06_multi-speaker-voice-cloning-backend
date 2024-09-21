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
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: linear-gradient(to right, #ff7e5f, #feb47b);
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            h1 {
                font-size: 3em;
            }
            img {
                max-width: 300px;
                margin-top: 20px;
                border-radius: 10px;
            }
            .icon {
                font-size: 2em;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>Chào mừng đến với trang của tôi!</h1>
            <div class="icon">😊</div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
