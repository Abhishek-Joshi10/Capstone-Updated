from flask import Flask, redirect, url_for, render_template, request, session, flash
from tensorflow.keras.models import Sequential, load_model
import pickle
from nltk.tokenize import RegexpTokenizer
import numpy as np
import heapq
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# model = pickle.load(open('NextWord.pkl', 'rb'))
model = load_model('next_word_model_v6.h5')

path = 'woman_clothing_reviews.txt'
text = open(path, encoding="utf8").read().lower()

# Tokenization

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

# Getting unique words

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

LENGTH_WORD = 5


# Testing Next Word
def prepare_input(text):
    x = np.zeros((1, LENGTH_WORD, len(unique_words)))
    for t, word in enumerate(text.split()):
        #        print(word)
        try:
            x[0, t, unique_word_index[word]] = 1
        except:
            pass
    return x


def sample(preds, top_n):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completions(text, n):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]


app = Flask(__name__)
app.secret_key = "esra"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class users(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    user = db.Column(db.String(100))

    def __init__(self, user):
        self.user = user


class history(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer())
    date = db.Column(db.DateTime, default=datetime.utcnow)
    text = db.Column(db.String(100))
    choice = db.Column(db.String(100))

    def __init__(self, user_id, date, text, choice):
        self.user_id = user_id
        self.date = date
        self.text = text
        self.choice = choice


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        username_form = request.form["username"]
        session["user"] = username_form

        found_user = users.query.filter_by(user=username_form).first()

        if found_user:
            #            session["userID"] = found_user.id
            flash("Login Successfull!")
        else:
            usr = users(username_form)
            db.session.add(usr)
            db.session.commit()
            found_user = users.query.filter_by(user=username_form).first()
        #            session["userID"] = found_user.id

        return redirect(url_for("main", text="a"))
    else:
        #        if "user" in session:
        #            flash("Already logged in!")
        return render_template("index.html")


@app.route("/main/<text>", methods=["POST", "GET"])
def main(text):
    if "user" in session:
        user = session["user"]
    if request.method == "POST":
        input_text = request.form["text"]
        x = prepare_input(input_text)
        preds = model.predict(x, verbose=0)[0]
        next_indices = sample(preds, 10)
        suggestions = predict_completions(input_text, 10)
        # string = suggestions[0] + ', ' + suggestions[1] + ', ' + suggestions[2] + ', ' + suggestions[3] + ', ' + suggestions[4] + ', ' + suggestions[5] + ', ' + suggestions[6] + ', ' + suggestions[7] + ', ' + suggestions[8] + ', ' + suggestions[9]
        return redirect(
            url_for("results", text=input_text, username=user, suggestions1=suggestions[0], suggestions2=suggestions[1],
                    suggestions3=suggestions[2], suggestions4=suggestions[3], suggestions5=suggestions[4]))
    else:
        flash("Login Successfull!")
        return render_template("main.html", username=user, text=text)


@app.route("/view")
def view():
    if "user" in session:
        user = session["user"]
    return render_template("view.html", values=users.query.all(), username=user)


@app.route("/user_history")
def user_history():
    if "user" in session:
        user = session["user"]
    found_user = users.query.filter_by(user=user).first()
    return render_template("user_history.html", values=history.query.filter_by(user_id=found_user.id).all(),
                           username=user)


@app.route("/results/<text>/<username>/<suggestions1>/<suggestions2>/<suggestions3>/<suggestions4>/<suggestions5>",
           methods=["POST", "GET"])
def results(text, username, suggestions1, suggestions2, suggestions3, suggestions4, suggestions5):
    if request.method == "POST" and request.form['suggestions']:
        updated_text = text + " " + request.form['suggestions']

        # saving user choice in history.db
        found_user = users.query.filter_by(user=username).first()
        curr_date = datetime.now()
        history_input = history(found_user.id, curr_date, text, request.form['suggestions'])
        db.session.add(history_input)
        db.session.commit()

        return redirect(url_for("main", text=updated_text))
    else:
        return render_template("results.html", text=text, username=username, suggestions1=suggestions1,
                               suggestions2=suggestions2, suggestions3=suggestions3, suggestions4=suggestions4,
                               suggestions5=suggestions5)


if __name__ == '__main__':
    db.create_all()
    app.run()