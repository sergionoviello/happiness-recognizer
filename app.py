import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import time
import re
from happiness_recognizer import HappinessRecognizer
from db_manager import DBManager

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
      # check if the post request has the file part
      if 'file' not in request.files:
          flash('No file part')
          return redirect(request.url)
      file = request.files['file']
      # if user does not select file, browser also
      # submit a empty part without filename
      if file.filename == '':
          flash('No selected file')
          return redirect(request.url)
      if file and allowed_file(file.filename):
          filename = secure_filename(file.filename)

          filename_w_prefix = ''.join([str(time.time()), '_', filename.lower()])

          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_w_prefix))
          return redirect(url_for('uploaded_file',
                                  filename=filename_w_prefix))


  sql_lite_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'db.sqlite')
  dbm = DBManager(sql_lite_file)

  happy = dbm.load_results(1)
  unhappy = dbm.load_results(0)

  return render_template('home.html', title='Happiness Recognizer', happy=happy, unhappy=unhappy)


@app.route('/results/<filename>')
def uploaded_file(filename):
  img = os.path.join(app.config['UPLOAD_FOLDER'], filename)

  happiness = HappinessRecognizer(img)
  result_txt, result = happiness.predict()

  sql_lite_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'db.sqlite')
  dbm = DBManager(sql_lite_file)
  dbm.save_result(filename, result)

  return render_template('results.html', result=result_txt, img=img)


