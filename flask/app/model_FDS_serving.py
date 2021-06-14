from flask_restful import Resource, reqparse, request
import json

class AiRun(Resource):
    def predict_request(self, data):
        ##Tensorflow Serving을 통해 예측 하는 로직 (request data 전처리 포함)
        return data

    def post(self):
        ##요청값을 받아오고 결과값을 주기 위한 로직

        #request를 파싱 하는 부분
        parser = reqparse.RequestParser()
        #type을 key로 한 값에 대해서 string 명시
        parser.add_argument('type',type=str)
        #data를 key로 한 값에 대해서 list형태 명시
        parser.add_argument('data', action='append', type=float)
        parser.add_argument('threshold',type=float)
        #파싱하여 args에 할당
        args = parser.parse_args()

        return args