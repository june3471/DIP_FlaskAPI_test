# About Flask and Flask_restful Library
from flask import Flask, render_template,request
from flask_restful import Api
from model_FDS_serving import AiRun

app = Flask(__name__)
api = Api(app)

#api.add_resource(<python 객체>,<api path : string>) 

#REST api는 url 뒤에 /api/predict 과 같이 명시함
#REpresentational State Transfer API

#자원을 이름(자원의 표현)으로 구분하여 해당 자원의 상태(정보)를 주고 받는 모든 것을 의미
api.add_resource(AiRun,"/api/predict")

@app.route("/")
def hello():
    return render_template('/test.html')

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)