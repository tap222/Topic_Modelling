import pandas as pd
import numpy as np

df = pd.read_excel(r'In Scope Groups tickets.xlsx')

df['Description']=df['Summary*'].astype('str')+' '+df['Resolution'].astype('str')


print('-----------------------count_vect started----------------')

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word',       
                             min_df=10,   # minimum reqd occurences of a word 
                             stop_words='english',  # remove stop words
                             lowercase=True,        # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,  # max number of uniq words   
                            )

doc_term_matrix = count_vect.fit_transform(df['Description'].values.astype('U'))

print('-------------------doc_term_matrix created----------------')

print('-------------------LDA created----------------')

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=10,               # Number of topics
                                      max_iter=10,                   # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,               # Random state
                                      batch_size=128,                 # n docs in each learning iter
                                      evaluate_every = -1,            # compute perplexity every n iters, default: Don't
                                      n_jobs = -1                   # Use all available CPUs
                                     )
LDA.fit(doc_term_matrix)

print('-------------------LDA completed----------------')


import random

for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])
	
first_topic = LDA.components_[0]

top_topic_words = first_topic.argsort()[-10:]

for i in top_topic_words:
    print(count_vect.get_feature_names()[i])

f= open("topic_list.txt","w+")	
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')
	f.write("This is line %d\r\n" % [count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
	
	
topic_values = LDA.transform(doc_term_matrix)
topic_values.shape


print('-------------------Topic Prediction Started----------------')

df['Topic'] = topic_values.argmax(axis=1)

df.head()

print('-------------------Topic Prediction completed----------------')

df.to_excel('output_Topic.xlsx',index=False)