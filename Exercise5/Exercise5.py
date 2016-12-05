from flask import Flask
from flask import render_template
from flask import request
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

with open('clf1.pickle', 'rb') as f:
    classificator = pickle.load(f)
	
with open('count_vect1.pickle', 'rb') as f:
    count_vect = pickle.load(f)
    
with open('transformer1.pickle', 'rb') as f:
    tfidf_transformer = pickle.load(f)
	
	
app = Flask(__name__)

@app.route('/main', methods=['GET','POST'])
def main():
    if request.method == 'GET':
        return render_template("flask_template.html")
    if request.method == 'POST':
        text = request.form['input']
        counts = count_vect.transform([text])
        transformered = tfidf_transformer.transform(counts)
        predicted = classificator.predict(transformered)
        return render_template("flask_template.html", text=text, predicted=predicted[0])


if __name__ == '__main__':
	app.debug = True
	app.run()