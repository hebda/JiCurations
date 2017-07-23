from flask import render_template, request
from app import app
import pymysql as mdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import fnmatch
import collections
from datetime import datetime
from sys import version_info

# for determining if email input box is needed
is_subscribed=0

#consider using a utils file
class product:
    def __init__(self,code,title,desc,features,price,imagecode):
        self.code=code
        self.title=title
        self.desc=desc
        self.features=features
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

@app.route('/yoga_product')
@app.route('/yoga_mat_bags')
def yoga_mat_bags():
    input_code=request.args.get('idd')
    products=[]
    product_i=None
    with open('app/data/yoga_mat_bags.csv') as filehandle:
        all_lines=''.join(filehandle.readlines())+'\n'
        for line in all_lines.split(',la fin')[1:-1]:
            code=line.split(',')[0].replace('\n','').replace('\r','')
            title=line.split(',')[1].replace('"','')
            desc=','.join([str(i) for i in line.split(',')[2:len(line.split(','))-3]]).replace('"','')
            if version_info.major<3:
                desc=desc.decode('utf-8')
            features=line.split(',')[-3].replace('"','').replace('\n','</li><li>').replace(':</li>',':<ul align="left" style="margin-top: -18px;">')+'</ul>'
            if version_info.major<3:
                features=features.decode('utf-8')
            price=int(line.split(',')[-2])
            imagecode=line.split(',')[-1].replace('\n','')
            products.append(product(code,title,desc,features,price,imagecode))
            if input_code==code:
                product_i=product(code,title,desc,features,price,imagecode)
                print(desc,"hi!!!",features)
    payload_dict={'products':products,'banner':'img.jpg','desc':'description'}
    with open('app/data/section_header.csv') as filehandle:
        for line in filehandle:
            if line.split(',')[0]=='yoga mat bags':
                payload_dict['banner']=line.split(',')[1]
                payload_dict['desc']=','.join(line.split(',')[2:]).replace('"','')
                break
    if input_code==None:
        return render_template('product_section.html',is_subscribed=is_subscribed,payload=payload_dict)
    else:
        num_img=len(fnmatch.filter(os.listdir("app/static/img"), '%s*'%input_code))
        return render_template('product.html',is_subscribed=is_subscribed,product_i=product_i,num_img=num_img)

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

