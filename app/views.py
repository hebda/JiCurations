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

# for tracking number of items and list of items in cart
num_items_in_cart=0
items_in_cart={}

# consider using a utils file
class product:
    def __init__(self,code,title,desc,features,price,imagecode):
        self.code=code
        self.title=title
        self.desc=desc
        self.features=features
        self.price=price
        self.imagecode=imagecode

# process products one time
product_sections={}
products={}
with open('app/data/section_header.csv') as filehandle:
    for line_num,line in enumerate(filehandle):
        if line_num==0:
            continue
        product_sections[line.split(',')[0]]={'banner':line.split(',')[1],
                                              'desc':','.join(line.split(',')[2:]).replace('"',''),
                                              'codes':[] }
for product_section_i in product_sections:
    with open('app/data/%s.csv'%product_section_i) as filehandle:
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
            products[code]=product(code,title,desc,features,price,imagecode)
            product_sections[product_section_i]['codes'].append(code)

font = {'family' : 'sans-serif', 'weight' : 'bold', 'size'   : 22}
plt.rc('font', **font)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

@app.route('/')
@app.route('/index')
@app.route('/input')
def home():
    return render_template("index.html",is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart)

@app.route('/coming_soon')
def coming_soon():
    return render_template('coming_soon.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart)

@app.route('/basket')
def basket():
    input_code=request.args.get('idd')
    if input_code!=None:
        if input_code in items_in_cart:
            items_in_cart[input_code]+=1
        else:
            items_in_cart[input_code]=1
        num_items_in_cart=sum(items_in_cart.values())
    return render_template('basket.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,items_in_cart=items_in_cart,products=products)

@app.route('/product')
def display_product():
    input_code=request.args.get('idd')
    product_i=products[input_code]
    num_img=len(fnmatch.filter(os.listdir("app/static/img"), '%s*'%input_code))
    return render_template('product.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,product_i=product_i,num_img=num_img)

@app.route('/product_section')
def display_product_section():
    input_code=request.args.get('idd')
    product_list=[]
    for code_i in product_sections[input_code]['codes']:
        product_list.append(products[code_i])
    payload_dict={'products':product_list,'banner':'img.jpg','desc':'description'}
    payload_dict['banner']=product_sections['yoga_mat_bags']['banner']
    payload_dict['desc']=product_sections['yoga_mat_bags']['desc']
    return render_template('product_section.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,payload=payload_dict)

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

    return render_template('index.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart)

@app.route('/error')
def error():
    return render_template('error.html')

