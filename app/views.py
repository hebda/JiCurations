from flask import render_template, request
from app import app
import pymysql as mdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import collections

font = {'family' : 'sans-serif', 'weight' : 'bold', 'size'   : 22}
plt.rc('font', **font)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

@app.route('/')
@app.route('/index')
@app.route('/input')
def git_input():

    return render_template("index.html")

@app.route('/coming_soon')
def slides():
    return render_template('coming_soon.html')

@app.route('/error')
def error():
    return render_template('error.html')

