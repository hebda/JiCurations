from flask import render_template, request
from app import app
import pymysql as mdb
import matplotlib.pyplot as plt
import math
import os
import collections
#from a_Model import ModelIt

month='October'

#the top languages from repo
languages=[
'C',
'C#',
'C++',
'Clojure',
'CoffeeScript',
'CSS',
'Go',
'Haskell',
'HTML',
'Java',
'Javascript',
'Lua',
'Objective-C',
'PHP',
'Perl',
'Python',
'R',
'Ruby',
'Rust',
'Scala',
'Shell',
'Swift',
#'TeX',
'TypeScript',
'VimL']

font = {'family' : 'sans-serif', 'weight' : 'bold', 'size'   : 22}
plt.rc('font', **font)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

@app.route('/')
@app.route('/index')
@app.route('/input')
def git_input():

    return render_template("index.html", month=month, languages=languages, hotness=1)

@app.route('/output_list')
@app.route('/output')
def output_list():
  #pull 'name' from input field and store it
  description_tag = request.args.get('description_tag')
  hipsterness = request.args.get('hipsterness')
  hotness = request.args.get('hotness')
  language_list=[]
  for i in languages:
      if request.args.get(i)==i:
          language_list.append(i)

  display_num=10

  db = mdb.connect(user="root", host="localhost", db="GitWatch", charset='utf8')
  with db:
    cur = db.cursor()

    where_clause="WHERE status=1 AND pred1 IS NOT NULL and language!=\"\""
    if hipsterness=='hipsterness':
        where_clause+=" AND hipster=1"
    if len(language_list)>0:
        where_clause+=" AND (language=\"%s\"" % language_list[0]
        for i in language_list[1:]:
            where_clause+=" OR language=\"%s\"" % i
        where_clause+=")"

    cur.execute("SELECT id,name,description,language,watchers,pred1,pred2,hipster FROM repo %s ORDER BY pred1 DESC;" % where_clause )

    result_list=list(cur.fetchall())
    watchers_list=[]
    for i in range(len(result_list)):
        result_list[i]=list(result_list[i])
        #result_list[i][1]=result_list[i][1][result_list[i][1].find('/')+1:]
        result_list[i][2]=result_list[i][2].replace('?','')
        if len(result_list[i][2])>160:
            result_list[i][2]=result_list[i][2][:160]+' ...'
        watchers_list.append(result_list[i][6])

    watchers_list=sorted(watchers_list)
    hotness_definition=watchers_list[int(math.floor(len(watchers_list)*0.9))]


    for i in range(len(result_list)):
        if result_list[i][6]>=hotness_definition:
            result_list[i].append(1)
        else:
            result_list[i].append(0)

    if hotness=='hotness':
        tmp_list=[]
        for i in range(len(result_list)):
            if result_list[i][8]:
                tmp_list.append(result_list[i])
                if len(tmp_list)==display_num:
                    break
        result_list=tmp_list
    else:
        result_list=result_list[:display_num]


    monthly_results=[]
    for i in result_list:
        repoid=i[0]

        fname_push='app/static/img/plots/%d_push.png' % repoid
        fname_watch='app/static/img/plots/%d_watch.png' % repoid
        if os.path.isfile(fname_push) and os.path.isfile(fname_watch):
            continue

        cur.execute("SELECT CASE WHEN timestamp BETWEEN '2015-06-01' AND '2015-07-01' THEN 'June' WHEN timestamp BETWEEN '2015-07-01' AND '2015-08-01' THEN 'July' WHEN timestamp BETWEEN '2015-08-01' AND '2015-09-01' THEN 'August' WHEN timestamp BETWEEN '2015-09-01' AND '2015-10-01' THEN 'September' ELSE 'October' END AS month, count(type) AS num_events FROM event WHERE type=17 and id=%d GROUP BY month;" % repoid)
    
        hist_push = {}
        for result in cur.fetchall():
            hist_push[result[0]]=result[1]
        hist_push['Pred. Oct.']=i[5]
    
        cur.execute("SELECT CASE WHEN timestamp BETWEEN '2015-06-01' AND '2015-07-01' THEN 'June' WHEN timestamp BETWEEN '2015-07-01' AND '2015-08-01' THEN 'July' WHEN timestamp BETWEEN '2015-08-01' AND '2015-09-01' THEN 'August' WHEN timestamp BETWEEN '2015-09-01' AND '2015-10-01' THEN 'September' ELSE 'October' END AS month, count(type) AS num_events FROM event WHERE type=19 and id=%d GROUP BY month;" % repoid)
        hist_watch={}
        for result in cur.fetchall():
            hist_watch[result[0]]=result[1]
        hist_watch['Pred. Oct.']=i[6]
    
        od_push=collections.OrderedDict()
        od_watch=collections.OrderedDict()
        for month_i in ['June','July','August','September','Pred. Oct.']:
            if month_i not in hist_push:
                hist_push[month_i]=0
                od_push[month_i]=0
            else:
                od_push[month_i]=hist_push[month_i]
            if month_i not in hist_watch:
                hist_watch[month_i]=0
                od_watch[month_i]=0
            else:
                od_watch[month_i]=hist_watch[month_i]


        fig = plt.figure()
        fig.set_size_inches((7,5))
        fig.patch.set_alpha(0.75)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.75)

        fig.subplots_adjust(bottom=0.15)
        ax.bar(range(len(od_push)),od_push.values(),align='center',color='#4589C7',alpha=0.75)
        ax.set_xticks( range(len(od_push)) )
        ax.set_xticklabels( od_push.keys(), rotation=20 ) ;
        plt.title('Pushes for %s' % i[1][i[1].find('/')+1:].replace('_','\_') )
        fig.savefig(fname_push)

        fig = plt.figure()
        fig.set_size_inches((7,5))
        fig.patch.set_alpha(0.75)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.75)

        fig.subplots_adjust(bottom=0.15)
        ax.bar(range(len(od_watch)),od_watch.values(),align='center',color='#4589C7',alpha=0.75)
        ax.set_xticks( range(len(od_watch)) )
        ax.set_xticklabels( od_watch.keys(), rotation=20 ) ;
        plt.title('Watches for %s' % i[1][i[1].find('/')+1:] )
        fig.savefig(fname_watch)


  return render_template("index.html", month=month, the_result = result_list, display_list_result=1, languages=languages, language_checklist=language_list, hotness=hotness, hipsterness=hipsterness)

@app.route('/slides')
def slides():
    return render_template('slides.html')

@app.route('/error')
def error():
    return render_template('error.html')

