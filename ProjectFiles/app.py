#from  import Flask, render_template, request, jsonify, send_file
from flask import Flask, render_template, request, send_file
import image_processor
import pandas as pd
import os

app = Flask(__name__)

for file_name in os.listdir():
    if file_name.endswith(".xlsx"):
        os.remove(file_name)
        
@app.route('/', methods=['GET', 'POST'])
def index():
    filename = "output.xlsx"
    if os.path.exists(filename):
        data = pd.read_excel(filename)
    else:
        data = pd.DataFrame()

    if request.method == 'POST':
        image = request.files['image'].read()
        data_new = image_processor.image_process(image)
        data = pd.concat([data, data_new], ignore_index=True)
        data.to_excel(filename, index=False)

    data_dict = data.to_dict(orient='records')
    return render_template('index.html', data=data_dict, filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        return "File not found."

if __name__ == '__main__':
    app.run()
