#!flask/bin/python
import os
from flask import Flask, render_template, request, Markup
import csv
from sklearn import linear_model
from werkzeug.utils import secure_filename
from models import SupervisedBinaryClassificationModel, SupervisedEstimationModel

UPLOAD_FOLDER = os.getcwd() + '/datasets'
ALLOWED_EXTENSIONS = {'csv'}
training_set_ratio = 80
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_data(filename):
    with open(filename, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        raw_data = [data for data in data_iter]
        return raw_data


def create_html_table(predictions, model, referrer):
    table = '<table class="table table-bordered"><tr>'
    for column_name in model.column_names:
        table += '<th>' + column_name + '</th>'
    table += '</tr>'
    for tup in predictions:
        if referrer == 'classify':
            table += "<tr><td>" + str(model.classes[tup[0]]) + "</td>"
        elif referrer == 'estimate':
            table += "<tr><td>" + str(tup[0]) + "</td>"
        for feature in tup[1]:
            table += "<td>" + str(feature) + "</td>"
        table += "</tr>"
    print(referrer)
    if referrer == 'classify':
        table += '<tr><td> Precisión: ' + str(model.accuracy_metrics()) + '</td></tr>'
        table += '<tr><td> Puntaje F1: ' + str(model.calculate_f1_score()) + '</td></tr></table>'
    elif referrer == 'estimate':
        table += '<tr><td>Coeficiente R^2: ' + str(model.calculate_r2_score(model.x_test, model.y_test)) + '</td></tr></table>'
    return table


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route("/requirements")
def requirements():
    return render_template('requirements.html')


@app.route('/classify')
def classify_view():
    return render_template("classify_estimate.html")


@app.route('/estimate')
def estimate_view():
    return render_template("classify_estimate.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if ('train_test_set' or 'predict_set') not in request.files:
        return render_template("error.html", error="Uno de los archivos no fue definido.")
    train_test_set = request.files['train_test_set']
    predict_set = request.files['predict_set']
    if train_test_set.filename == '' or predict_set.filename == '':
        return render_template("error.html", error="Uno de los archivos no fue definido.")
    if (train_test_set and allowed_file(train_test_set.filename)) \
        and (predict_set and allowed_file(predict_set.filename)):
        train_test_set_filename = secure_filename(train_test_set.filename)
        train_test_set_path = os.path.join(app.config['UPLOAD_FOLDER'], train_test_set_filename)
        train_test_set.save(train_test_set_path)
        predict_set_filename = secure_filename(predict_set.filename)
        predict_set_path = os.path.join(app.config['UPLOAD_FOLDER'], predict_set_filename)
        predict_set.save(predict_set_path)
        referrer = request.referrer
        function = referrer[referrer.rfind('/') + 1:]
        raw_training_data = load_data(train_test_set_path)
        raw_prediction_data = load_data(predict_set_path)

        if function == 'classify' or function == 'estimate':
            model = None
            if function == 'classify':
                model = SupervisedBinaryClassificationModel(raw_training_data, raw_prediction_data, \
                                                            linear_model.LogisticRegression())
            elif function == 'estimate':
                model = SupervisedEstimationModel(raw_training_data, raw_prediction_data, \
                                                  linear_model.LinearRegression())
            model.train()
            predictions = model.predict()
            result = create_html_table(predictions, model, function)
        else:
            return render_template("error.html", error="Se intentó realizar una acción inválida.")
        return render_template("results.html", result=Markup(result))
    return render_template("error.html", error="No está permitido ese formato de archivo.")


if __name__ == '__main__':
    app.run(debug=True)
