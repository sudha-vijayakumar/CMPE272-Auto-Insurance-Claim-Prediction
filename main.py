from flask import Flask,jsonify
import joblib 
import pandas
import json
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask import request
import math
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/predict_claim', methods=['GET', 'POST'])
def predict_claim():

    data = request.get_json(force=True)

    index = [1]
    cust_df = pandas.DataFrame(data,index)

    # Load the model from the file 
    trained_model = RandomForestRegressor()
    trained_model = joblib.load('trained_model.pkl')  
    
    # Load the pickled model 
    prediction_cust= trained_model.predict(cust_df)


    return jsonify({'predicted claim': str(prediction_cust[0])})
	

@app.route("/get_similar_claims", methods=['GET','POST'])
def get_similar_claims():
    
    data = request.get_json(force=True)

    #read encoded insurance data.
    df = pandas.read_csv('encoded.csv')

    df.set_index("claim_id", drop=True, inplace=True)

    #remove values that cannot be encoded.
    del df['join_date']
    del df['income']
    del df['monthly_premium']
    del df['months_since_last_claim']
    del df['months_since_policy_inception']
    del df['open_complaints']
    del df['no_of_policies']
    del df['total_claim_amount']
    del df['vehicle_no']
    del df['policy_level']
    del df['Unnamed: 0']
    del df['first_name']
    del df['last_name']
    del df['email']

    #format data to find similar claims.
    dictionary = df.to_dict(orient="index")

    #claimID to query.
    claimID = data.get('claim_id')
    
    #No.of.similar claims to query.
    cnt_similar_claims = data.get('cnt_similar_claims')
    similar_claims = calculateSimilarItems(dictionary,claimID,cnt_similar_claims)

    return jsonify({'predicted claim': str(similar_claims)})

#function calculates distance start
def euclidean_distnce(data, p1, p2):
    common_item = {}
    for item in data[p1]:
        if item in data[p2]:
            common_item[item] = True

    if len(common_item) == 0: return 0

    #calculate Euclidean distance
    #√((x1-x2)^2 + (y1-y2)^2)
    distance = sum([math.pow(data[p1][itm] - data[p2][itm], 2) for itm in common_item.keys()])
    distance = math.sqrt(distance)
    #return result
    return 1/(distance + 1)

def calculateSimilarItems(data, claimId, n):   
    result = {}
    data_reverse = data 
    
    item = claimId
    
    #finding distance score of all other items with respect to current item
    similarities = [(euclidean_distnce(data_reverse, item, other), other) for other in data_reverse.keys() if item == claimId] 
    similarities.sort()   
    result[item] = similarities[0:n]   
    return result

if __name__ == '__main__':
	app.run(host="127.0.0.1",port=8080,debug=True)