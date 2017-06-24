from flask import render_template, request
from app import app
import pymysql as mdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import collections
from datetime import datetime

font = {'family' : 'sans-serif', 'weight' : 'bold', 'size'   : 22}
plt.rc('font', **font)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

# for determining if email input box is needed
is_subscribed=0

@app.route('/')
@app.route('/index')
@app.route('/input')
def home():
    return render_template("index.html",is_subscribed=is_subscribed)

@app.route('/coming_soon')
def coming_soon():
    return render_template('coming_soon.html',is_subscribed=is_subscribed)

@app.route('/subscribe')
def subscribe():
    email_input=request.args.get('email_input')
    dateint=int(datetime.utcnow().strftime('%Y%m%d'))
    if '@' in email_input and '.' in email_input:
        is_subscribed=1
        db = mdb.connect(user="root", host="localhost", db="JiCurations", charset='utf8')
        with db:
            cur = db.cursor()
            cur.execute("INSERT INTO subscription (email, subscription_date) values (\"%s\",%d);" % (email_input,dateint) )

    return render_template('index.html',is_subscribed=is_subscribed)

@app.route('/error')
def error():
    return render_template('error.html')

