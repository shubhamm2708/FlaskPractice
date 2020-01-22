from surprise import KNNBasic
import pandas as pd
from surprise import Dataset
from surprise import Reader

colleges = ['TCET', 'TCET', 'SFIT', 'SFIT', 'ACOE', 'ACOE', 'TCET', 'TIMSR']
rating = [1, 5, 2, 1, 5, 3, 2, 5]
users = ['Kim', 'Tim', 'John', 'Jimmy', 'Julia', 'Kim', 'Jimmy', 'Kim']
x = []
rating_dict = {'users' : users,
               'colleges' : colleges,
               'rating' : rating}

def FriendRecommender(user):
    df = pd.DataFrame(rating_dict)
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[['users', 'colleges', 'rating']], reader)
    trainset = data.build_full_trainset()
    sim_options = {
        'name' : 'cosine',
        'user-based' : True 
    }
    
    
    algo = KNNBasic(sim_options)
    algo.fit(trainset)
    
    uid = trainset.to_inner_uid(user)
    pred = algo.get_neighbors(uid, 3)
    
    for i in pred:
        x.insert(i, (trainset.to_raw_uid(i)))
        
        
FriendRecommender('Tim')