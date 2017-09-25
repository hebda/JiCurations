from flask import render_template, request, session
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
import hashlib
from random import randint

# for determining if email input box is needed
is_subscribed=0

# for tracking number of items and list of items in cart
num_items_in_cart=0
items_in_cart={}
cart_total=0
order_total=0

# secret key
with open('app/data/key_and_salt.txt') as filehandle:
    for line_num,line in enumerate(filehandle):
        if line_num==0:
            continue
        elif line_num==1:
            tmp_salt=line.replace('\n','')
        if line_num>=2:
            break
app.secret_key=os.urandom(int(hashlib.sha256((tmp_salt+str(datetime.utcnow())).encode('utf-8')).hexdigest()[:2],16))
app.secret_key='hi how'
app.config['SECRET_KEY']='hi how'

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

def calc_tax(subtotal=0):
    tax_rate=0.0
    return int(subtotal*tax_rate)

def calc_shipping(subtotal=0):
    return 50 if subtotal>0 and subtotal<1000 else 0

def get_session_object(object_,default_val):
    if object_ in session:
        return session[object_]
    else:
        return default_val

@app.route('/')
@app.route('/index')
@app.route('/input')
def home():
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    is_subscribed=get_session_object('is_subscribed',0)
    return render_template("index.html",is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart)

@app.route('/coming_soon')
def coming_soon():
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    is_subscribed=get_session_object('is_subscribed',0)
    return render_template('coming_soon.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart)

@app.route('/update_basket')
def update_basket():
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    items_in_cart=get_session_object('items_in_cart',{})
    cart_total=get_session_object('cart_total',0)
    order_total=get_session_object('order_total',0)
    is_subscribed=get_session_object('is_subscribed',0)

    input_code=request.args.get('id_zero')
    if input_code!=None:
        if input_code in items_in_cart:
            del items_in_cart[input_code]
    for arg_i in request.args:
        if 'value_' not in arg_i:
            continue
        val_i=int(request.args.get(arg_i))
        if arg_i[6:] in products:
            if val_i<=0:
                del items_in_cart[arg_i[6:]]
            else:
                items_in_cart[arg_i[6:]]=val_i
        
    num_items_in_cart=sum(items_in_cart.values())
    cart_total=sum([items_in_cart[i]*products[i].price for i in items_in_cart])
    session['num_items_in_cart']=num_items_in_cart
    session['items_in_cart']=items_in_cart
    session['cart_total']=cart_total

    return basket()

@app.route('/basket')
def basket():
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    items_in_cart=get_session_object('items_in_cart',{})
    cart_total=get_session_object('cart_total',0)
    order_total=get_session_object('order_total',0)
    is_subscribed=get_session_object('is_subscribed',0)

    input_code=request.args.get('idd')
    if input_code!=None:
        if input_code in items_in_cart:
            items_in_cart[input_code]+=1
        else:
            items_in_cart[input_code]=1
        num_items_in_cart+=1
        cart_total+=products[input_code].price
    tax_total=calc_tax(cart_total)
    shipping_total=calc_shipping(cart_total)
    order_total=cart_total+tax_total+shipping_total

    session['num_items_in_cart']=num_items_in_cart
    session['items_in_cart']=items_in_cart
    session['cart_total']=cart_total
    session['order_total']=order_total

    return render_template('basket.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,items_in_cart=items_in_cart,products=products,cart_total=cart_total,tax_total=tax_total,shipping_total=shipping_total,order_total=order_total)

@app.route('/checkout', methods=['GET', 'POST'])
def checkout():
    is_testing=1
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    items_in_cart=get_session_object('items_in_cart',{})
    cart_total=get_session_object('cart_total',0)
    order_total=get_session_object('order_total',0)
    is_subscribed=get_session_object('is_subscribed',0)

    tax_total=calc_tax(cart_total)
    shipping_total=calc_shipping(cart_total)

    MERCHANT_KEY = ""
    SALT = ""
    with open('app/data/key_and_salt%s.txt' % ('_test' if is_testing else '') ) as filehandle:
        for line_num,line in enumerate(filehandle):
            if line_num==0:
                MERCHANT_KEY=line.replace('\n','')
            elif line_num==1:
                SALT=line.replace('\n','')
            if line_num>=2:
                break
    PAYU_BASE_URL = "https://secure.payu.in/_payment"
    if is_testing:
        PAYU_BASE_URL = "https://test.payu.in/_payment"
    action=PAYU_BASE_URL
    posted={}
    hash_object = hashlib.sha256((str(datetime.utcnow())+str(randint(0,2000))).encode('utf-8'))
    txnid=hash_object.hexdigest()[0:20]
    hashh = ''
    posted['txnid']=txnid
    hashSequence = "key|txnid|amount|productinfo|firstname|email|udf1|udf2|udf3|udf4|udf5|udf6|udf7|udf8|udf9|udf10"
    posted['key']=MERCHANT_KEY
    posted['amount']=order_total
    posted['productinfo']='; '.join(['%s:%d'%(i,items_in_cart[i]) for i in items_in_cart])
    posted['country']='India'
    posted['surl']='http://jicurations.com/product_section?idd=yoga_mat_bags'
    posted['furl']='jicurations.com'
    posted['firstname']=''
    posted['email']='' if not is_testing else 'email'
    hash_string=''
    for i in request.form:
        posted[i]=request.form[i]
    hashVarsSeq=hashSequence.split('|')
    for i in hashVarsSeq:
        try:
            hash_string+=str(posted[i])
        except Exception:
            hash_string+=''
        hash_string+='|'
    hash_string+=SALT
    hashh=hashlib.sha512(hash_string.encode('utf-8')).hexdigest().lower()

    if request.args.get('id')=='edit':
        return render_template('checkout1.html',posted=posted,hashh=hashh,hash_string=hash_string,action=action,is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,items_in_cart=items_in_cart,products=products,cart_total=cart_total,tax_total=tax_total,shipping_total=shipping_total,order_total=order_total)
    elif request.args.get('id')=='verify' and posted['firstname']!='' and posted["email"]!='':
        return render_template('checkout2.html',posted=posted,hashh=hashh,hash_string=hash_string,action=action,is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,items_in_cart=items_in_cart,products=products,cart_total=cart_total,tax_total=tax_total,shipping_total=shipping_total,order_total=order_total)
    elif request.args.get('id')=='verify' and (posted['firstname']=='' or posted["email"]==''):
        return render_template('checkout1.html',posted=posted,hashh=hashh,hash_string=hash_string,action=action,is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,items_in_cart=items_in_cart,products=products,cart_total=cart_total,tax_total=tax_total,shipping_total=shipping_total,order_total=order_total,error=1)
    else:
        return render_template('checkout1.html',posted=posted,hashh=hashh,hash_string=hash_string,action=action,is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,items_in_cart=items_in_cart,products=products,cart_total=cart_total,tax_total=tax_total,shipping_total=shipping_total,order_total=order_total)


@app.route('/product')
def display_product():
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    is_subscribed=get_session_object('is_subscribed',0)

    input_code=request.args.get('idd')
    product_i=products[input_code]
    num_img=len(fnmatch.filter(os.listdir("app/static/img"), '%s*'%input_code))
    return render_template('product.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart,product_i=product_i,num_img=num_img)

@app.route('/product_section')
def display_product_section():
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    is_subscribed=get_session_object('is_subscribed',0)

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
    num_items_in_cart=get_session_object('num_items_in_cart',0)
    is_subscribed=get_session_object('is_subscribed',0)

    email_input=request.args.get('email_input')
    dateint=int(datetime.utcnow().strftime('%Y%m%d'))
    if '@' in email_input and '.' in email_input:
        is_subscribed=1
        db = mdb.connect(user="root", host="localhost", db="JiCurations", charset='utf8')
        with db:
            cur = db.cursor()
            cur.execute("INSERT INTO subscription (email, subscription_date) values (\"%s\",%d);" % (email_input,dateint) )

    session['is_subscribed']=is_subscribed

    return render_template('index.html',is_subscribed=is_subscribed,num_items_in_cart=num_items_in_cart)

@app.route('/error')
def error():
    return render_template('error.html')

