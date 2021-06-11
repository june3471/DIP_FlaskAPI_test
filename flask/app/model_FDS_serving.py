from flask_restful import Resource, reqparse, request
import json
import pickle
import numpy as np
import os
import requests


class AiRun(Resource):
    def predict_request(self, data):
        # Tensorflow Serving을 통해 예측 하는 로직 (request data 전처리 포함)
        """
        data : list 형태로 들어올 것 1차원 
        """
        # 모델 학습에 적용되었는 변수별 min max 값 정의
        col_max_value = [2.45492999121121,
                         22.0577289904909,
                         4.07916781154883,
                         16.8753440335975,
                         34.8016658766686,
                         73.3016255459646,
                         120.589493945238,
                         20.0072083651213,
                         10.370657984046,
                         15.2360282040071,
                         12.0189131816199,
                         7.8483920756446,
                         7.12688295859376,
                         10.5267660517847,
                         8.87774159774277,
                         17.3151115176278,
                         9.25352625047285,
                         4.71239756635225,
                         5.59197142733558,
                         38.1172091261285,
                         27.2028391573154,
                         10.5030900899454,
                         22.5284116897749,
                         4.58454913689817,
                         5.8261590349735,
                         3.41563624349633,
                         31.6121981061363,
                         33.8478078188831,
                         25691.16]
        col_min_value = [
            -56.407509631329,
            -72.7157275629303,
            -48.3255893623954,
            -5.68317119816995,
            -113.743306711146,
            -26.1605059358433,
            -43.5572415712451,
            -73.2167184552674,
            -13.4340663182301,
            -24.5882624372475,
            -4.79747346479757,
            -18.6837146333443,
            -5.79188120632084,
            -19.2143254902614,
            -4.49894467676621,
            -14.1298545174931,
            -25.1627993693248,
            -9.49874592104677,
            -6.93829731768481,
            -54.497720494566,
            -34.8303821448146,
            -9.49942296430251,
            -30.269720014317,
            -2.82484890293617,
            -7.08132534637739,
            -2.60455055280817,
            -9.89524404755692,
            -15.4300839055349,
            0.0]
        col_max_value = np.array(col_max_value).reshape(1, -1)
        col_min_value = np.array(col_min_value).reshape(1, -1)

        # MinMax Scale을 적용하여 모델 input value 형태로 변환
        data_new = np.array(data).reshape(1, -1).astype(float)
        data_new = (data_new - col_min_value)/(col_max_value - col_min_value)

        # List 형태 최종 model input value 로 변환
        instances = data_new.tolist()

        # Tensorflow Serving api input 형태인 json 으로 변환
        data = json.dumps(
            {"signature_name": "serving_default", "instances": instances})  # instances key에 예측 데이터를 value로 넣어야함
        headers = {"content-type": "application/json"}

        # Tensorflow Serving 컨테이녀  ip 주소 받아오기
        model_api_ip = "172.10.18.04"
        # Tensorflow Serving Model Name
        model_name = "FDSmodel"
        # Tensorflow Serving 컨테이너 api 주소
        model_api_url = "http://%s:8501/v1/models/%s:predict" % (
            model_api_ip, model_name)

        # Tensorflow Serving api post 해서 예측 결과 값 받아오기
        json_response = requests.post(
            model_api_url, data=data, headers=headers)
        pred_value = json_response.json()["predictions"]

        return pred_value, data_new

    def post(self):
        # 요청값을 받아오고 결과값을 주기 위한 로직
        try:
            # request를 파싱 하는 부분
            parser = reqparse.RequestParser()
            # threshold를 key로 한 값에 대해서 float형태 명시 - abnormal 관측치에 대한 판단 기준을 명시해주는 것
            parser.add_argument('threshold', type=float)
            # data를 key로 한 값에 대해서 list형태 명시
            parser.add_argument('data', action='append')
            # 파싱하여 args에 할당
            args = parser.parse_args()

            # Tenseroflow Serving 예측 결과 받아오기
            pred_value, data_new = self.predict_request(args["data"])
            print(pred_value)
            # 예측값 실제값 차이 계산
            loss = np.abs(np.array(pred_value) - np.array(data_new)).mean()
            # loss 가 Threshold 기준보다 클 경우 이상 관측치 1, 그렇지 않으면 정상 관측치 0
            if loss > args["threshold"]:
                pred_result = 1
            else:
                pred_result = 0
            return {"result": pred_result}
        except Exception as e:
            return {'error': str(e)}
