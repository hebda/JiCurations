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

# for determining if email input box is needed
is_subscribed=0

#consider using a utils file
class product:
    def __init__(self,code,title,desc,price,imagecode):
        self.code=code
        self.title=title
        self.desc=desc
        self.price=price
        self.imagecode=imagecode

font = {'family' : 'sans-serif', 'weight' : 'bold', 'size'   : 22}
plt.rc('font', **font)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

@app.route('/')
@app.route('/index')
@app.route('/input')
def home():
    return render_template("index.html",is_subscribed=is_subscribed)

@app.route('/coming_soon')
def coming_soon():
    return render_template('coming_soon.html',is_subscribed=is_subscribed)

@app.route('/yoga_mat_bags')
def yoga_mat_bags():
    products=[]
    with open('app/data/yoga_mat_bags.csv') as filehandle:
        all_lines=''.join(filehandle.readlines())+'\n'
        for line in all_lines.split(',la fin\n')[1:-1]:
            code=line.split(',')[0]
            title=line.split(',')[1].replace('"','')
            desc=','.join([str(i) for i in line.split(',')[2:len(line.split(','))-2]]).replace('"','')
            price=int(line.split(',')[-2])
            imagecode=line.split(',')[-1].replace('\n','')
            products.append(product(code,title,desc,price,imagecode))
    payload_dict={'products':products,'banner':'img.jpg','desc':'description'}
    with open('app/data/section_header.csv') as filehandle:
        for line in filehandle:
            if line.split(',')[0]=='yoga mat bags':
                payload_dict['banner']=line.split(',')[1]
                payload_dict['desc']=','.join(line.split(',')[2:]).replace('"','')
                break
    return render_template('product_section.html',is_subscribed=is_subscribed,payload=payload_dict)

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

