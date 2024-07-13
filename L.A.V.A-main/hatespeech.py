# importing packages
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')
import pickle
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# importing packages for Plotly visualizations
import plotly
from plotly import graph_objs
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
plotly.offline.init_notebook_mode()

# import NLP packages
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from yellowbrick.text import FreqDistVisualizer, TSNEVisualizer
from wordcloud import WordCloud
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# import modeling packages
from sklearn import utils, svm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

%reload_ext autoreload
%autoreload 2
import sys
sys.path.append("py/")
from utils import *
from config import keys
from preprocess import *


df = pd.read_csv("data/original/labeled_data.csv", index_col=0)
df.head()

# get dimensions of dataframe
df.shape

# display class distribution
hate = len(df[df['target'] == 0])
off = len(df[df['target'] == 1])
neu = len(df[df['target'] == 2])
dist = [
    graph_objs.Bar(
        x=["hate", "offensive", "neutral"],
        y=[hate, off, neu],
)]
plotly.offline.iplot({"data":dist, "layout":graph_objs.Layout(title="Class Distribution")})
# create hate and non-hate categories by combining offensive and neutral categories
df.target = df.target.replace([2], 1)
df.target = df.target.replace([0, 1], [1, 0])
df.target.value_counts()


# create visualization for new target variable distribution
def barplot(df, feature, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=feature, data=df, ax=ax)
    plt.title(title, fontsize=16)
    plt.xlabel("target", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    total = len(df.target)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size=12)
    fig.show()
    fig.savefig("images/target_distribution.png")

plt.style.use('seaborn')
barplot(df, 'target', 'Distribution of Target Variable')

df.tweet.head(20)

# create functions to count number of words in tweet
def num_of_words(df, col):
    df['word_ct'] = df[col].apply(lambda x: len(str(x).split(" ")))
    print(df[[col, 'word_ct']].head())

num_of_words(df, 'tweet')

# create visualization for word count distribution
df['word_ct'].iplot(
    kind='hist',
    bins=40,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Word Count Distribution')
# create function to ccount number of characters in a tweet
def num_of_chars(df, col):
    df['char_ct'] = df[col].str.len()
    print(df[[col, 'char_ct']].head())

num_of_chars(df, 'tweet')

# create visualization to display character count distribution
df['char_ct'].iplot(
    kind='hist',
    bins=100,
    xTitle='character count',
    linecolor='black',
    yTitle='count',
    title='Character Count Distribution')

# create function to calculate average word length and then average word length per tweet
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

def avg_word_length(df, col):
    df['avg_wrd'] = df[col].apply(lambda x: avg_word(x))
    print(df[[col, 'avg_wrd']].head())

avg_word_length(df, 'tweet')

# create visualization for average word length distribution
df['avg_wrd'].iplot(
    kind='hist',
    bins=60,
    xTitle='average word length',
    linecolor='black',
    yTitle='count',
    title='Average Word Length Distribution')

# create function to count number of hashtags per tweet
def hash_ct(df, col):
    df['hash_ct'] = df[col].apply(lambda x: len(re.split(r'#', str(x)))-1)
    print(df[[col, 'hash_ct']].head())

hash_ct(df, 'tweet')

# create visualization for displaying hashtag distribution 
df['hash_ct'].iplot(
    kind='hist',
    bins=100,
    xTitle='hashtags count',
    linecolor='black',
    yTitle='count',
    title='Number of Hashtags Distribution')

preprocess_tweets(df, 'tweet')
df.head()

# display first five rows of dataframe
df.head()

# separate dataframe into respective classes
hate = df[df.target == 1]
non_hate = df[df.target == 0]

# separate features from target variable for train_test_splt
X_h = hate.tweet
y_h = hate.target
X_nh = non_hate.tweet
y_nh = non_hate.target

# perform 75-25 training-validation split and 15-10 validation-testing split on dataset
X_h_tr, X_h_val, y_h_tr, y_h_val = train_test_split(X_h, y_h, test_size=0.25, random_state=42)
X_h_val, X_h_tt, y_h_val, y_h_tt = train_test_split(X_h_val, y_h_val, test_size=0.4, random_state=42)
X_nh_tr, X_nh_val, y_nh_tr, y_nh_val = train_test_split(X_nh, y_nh, test_size=0.25, random_state=42)
X_nh_val, X_nh_tt, y_nh_val, y_nh_tt = train_test_split(X_nh_val, y_nh_val, test_size=0.4, random_state=42)
# concatenate hate and non-hate dataframe to reform entire training dataset
X_tr = pd.concat((X_h_tr, X_nh_tr), ignore_index=True)
y_tr = pd.concat((y_h_tr, y_nh_tr), ignore_index=True)
train = pd.concat([X_tr, y_tr], axis=1)

# # remove brackets around the list to create a list of strings
train['tweet2'] = train.tweet.apply(lambda x: str(x)[1:-1]) 
train.head()

# concatenate hate and non-hate dataframes to reform entire validation dataset
X_val = pd.concat((X_h_val, X_nh_val), ignore_index=True)
y_val = pd.concat((y_h_val, y_nh_val), ignore_index=True)
val = pd.concat([X_val, y_val], axis=1)

# remove brackets around the list to create a list of string
val['tweet2'] = val.tweet.apply(lambda x: str(x)[1:-1]) 
val.head()

X_tt = pd.concat((X_h_tt, X_nh_tt), ignore_index=True)
y_tt = pd.concat((y_h_tt, y_nh_tt), ignore_index=True)
test = pd.concat([X_tt, y_tt], axis=1)

# remove brackets around the list to create a list of string
test['tweet2'] = test.tweet.apply(lambda x: str(x)[1:-1]) 
test.head()

# split back into minority and majority classes for visualizations
zero = train[train.target == 0]
one = train[train.target == 1]
# create list of tokens for 
zero_tokens = []
for index, row in zero.iterrows():
    zero_tokens.extend(row['tweet'])

one_tokens = []
for index, row in one.iterrows():
    one_tokens.extend(row['tweet'])
# convert collection of text documents to matrix of token counts
vec = CountVectorizer()

# learn vocabulary dictionary to return document-term matrix
docs = vec.fit_transform(zero_tokens)

# array mapping from feature integer indices to feature name
features = vec.get_feature_names()

# use Yellowbrick implementation of visualizing token frequency distribution
visualizer = FreqDistVisualizer(features=features, orient='h', n=25, size=(540, 360), color='tab:blue')
visualizer.fit(docs)
custom_viz = visualizer.ax
custom_viz.set_xlabel('Number of Tokens', fontsize=14)
custom_viz.set_ylabel('Token', fontsize=14)
custom_viz.set_title("Frequency Distribution of Top 25 Tokens for Non-Hate Tweets", fontsize=14)
custom_viz.figure.show()

# create visualization for positive class
vec_one = CountVectorizer()
docs_one = vec_one.fit_transform(one_tokens)
features_one = vec_one.get_feature_names()

visualizer_one = FreqDistVisualizer(features=features_one, orient='h', n=25, size=(540, 360), color='tab:orange')
visualizer_one.fit(docs_one)
custom_viz_one = visualizer_one.ax
custom_viz_one.set_xlabel('Number of Tokens', fontsize=14) 
custom_viz_one.set_ylabel('Token', fontsize=14)
custom_viz_one.set_title("Frequency Distribution of Top 25 Tokens for Hate Tweets", fontsize=14)
custom_viz_one.figure.show()

# create TSNE visualization for negative class
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(train.tweet2)
y = train.target

visualizer = TSNEVisualizer(alpha=0.1, colors=['lightgray', 'indigo'], decompose='svd', decompose_by=100, random_state=42)
visualizer.fit(X, y)
visualizer.show(outpath="images/tsne.png")
<AxesSubplot:title={'center':'TSNE Projection of 18586 Documents'}>

# create TSNE visualization for negative class
tfidf = TfidfVectorizer()
X_zero = tfidf.fit_transform(zero.tweet2)
y_zero = zero.target

visualizer = TSNEVisualizer(alpha=0.1, colors=['tab:blue'], decompose='svd', decompose_by=100, random_state=42)
visualizer.fit(X_zero, y_zero)
visualizer.show(outpath="images/tsne_zero.png")


# create TSNE visualization for negative class
tfidf = TfidfVectorizer()
X_one = tfidf.fit_transform(one.tweet2)
y_one = one.target

visualizer = TSNEVisualizer(alpha=0.6, decompose='svd', colors=['tab:orange'], decompose_by=100, random_state=42)
visualizer.fit(X_one, y_one)
visualizer.show(outpath="images/tsne_one.png")


text = ' '.join(zero_tokens)

# Initialize wordcloud object
wc = WordCloud(font_path="/Users/examsherpa/Library/Fonts/FiraMono-Medium.ttf", background_color='lightgray', colormap='tab10', max_words=50)

# Generate and plot wordcloud
plt.imshow(wc.generate(text))
plt.axis('off')
plt.show()

text = ' '.join(one_tokens)

# Initialize wordcloud object
wc = WordCloud(font_path="/Users/examsherpa/Library/Fonts/FiraMono-Medium.ttf", background_color='lightgray', colormap='tab10', max_words=50)

# Generate and plot wordcloud
plt.imshow(wc.generate(text))
plt.axis('off')
plt.show()

train.tweet2

# assign feature and target variables
X_tr = train.tweet2
X_val = val.tweet2
y_tr = train.target
y_val = val.target

# vectorize tweets for modeling
vec = TfidfVectorizer()
tfidf_tr = vec.fit_transform(X_tr)
tfidf_val = vec.transform(X_val)

nb = MultinomialNB().fit(tfidf_tr, y_tr)
y_pred_nb = nb.predict(tfidf_val)
get_metrics_confusion(tfidf_val, y_val, y_pred_nb, nb)

rf = RandomForestClassifier(n_estimators=100).fit(tfidf_tr, y_tr)
y_pred_rf = rf.predict(tfidf_val)
get_metrics_confusion(tfidf_val, y_val, y_pred_rf, rf)


log = LogisticRegression().fit(tfidf_tr, y_tr)
y_pred_log = log.predict(tfidf_val)
get_metrics_confusion(tfidf_val, y_val, y_pred_log, log)

svc = svm.LinearSVC(random_state=42).fit(tfidf_tr, y_tr)
y_pred_svc = svc.predict(tfidf_val)
get_metrics_2(tfidf_val, y_val, y_pred_svc, svc)


abc = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
    ).fit(tfidf_tr, y_tr)
y_pred_abc = abc.predict(tfidf_val)
get_metrics_confusion(tfidf_val, y_val, y_pred_abc, abc)

gbc = GradientBoostingClassifier().fit(tfidf_tr, y_tr)
y_pred_gbc = gbc.predict(tfidf_val)
get_metrics_confusion(tfidf_val, y_val, y_pred_gbc, gbc)

# display metrics from training
metrics = pd.read_csv("data/metrics/metrics.csv", index_col=0)
metrics

# display metrics for training after applying RUS
metrics2 = pd.read_csv("data/metrics/metrics2.csv", index_col=0)
metrics2

# display metrics from training after applying CNN
metrics3 = pd.read_csv("data/metrics/metrics3.csv", index_col=0)
metrics3

# display metrics from training after running SMOTE-ENN
metrics4 = pd.read_csv("data/metrics/metrics4.csv", index_col=0)
metrics4

# upload dataset
df = pd.read_csv('data/original/hatespeechtwitter.csv')
df.head()

# get value counts for target variable categories
df.columns = ['id', 'label']
df.label.value_counts()

# get tweet IDs into form to insert into API
df_hateful = df[df['label']=='hateful']
hate_ids = group_list(list(df_hateful.id))
len(hate_ids)

# setup url and headers for calling Twitter API as per documentation
url = "https://api.twitter.com/2/tweets?ids=847661947159891972,847799130277675008,848933211375779840&tweet.fields=created_at,entities,geo,id,public_metrics,text&user.fields=description,entities,id,location,name,public_metrics,username"
payload={}
headers = {'Authorization': 'Bearer ' + keys['bearer_token'], 'Cookie': 'personalization_id="v1_hzpv7qXpjB6CteyAHDWYQQ=="; guest_id=v1%3A161498381400435837'}

# make GET request to Twitter API to get response object 
r = requests.request("GET", url, headers=headers, data=payload)

# convert payload to dictionary
data = r.json()
# run function to download twitter text from Twitter API
df_hate = tweets_request(hate_ids)
# upload saved pickle to get dataframe
df_1 = pickle.load(open("pickle/aristotle_hate.pickle", "rb"))
df_1['target'] = 1
preprocess_tweets(df_1, 'text')
df_1 = df_1[['text', 'target']]
df_1.head()


len(df_1)

# upload dataset
df2 = pd.read_csv('data/original/NAACL_SRW_2016.csv')
df2.columns = ['id', 'label']

# get counts for the target variable
df2.label.value_counts()

# upload pickle to get dataframe
df_2 = pickle.load(open("pickle/copenhagen_2.pickle", "rb"))
df_2['target'] = 1
preprocess_tweets(df_2, 'text')
df_2 = df_2[['text', 'target']]
df_2.head()


len(df_2)

# upload dataset and display dataframe
english = pd.read_csv("data/original/english_dataset.tsv", delimiter="\t")
english.task_2.value_counts()

df_3 = english[english['task_2'] == 'HATE']
df_3['target'] = 1
preprocess_tweets(df_3, 'text')
df_3 = df_3[['text', 'target']]
df_3.head()

# upload dataset and display dataframe
df_4 = pd.read_csv("data/original/labeled_data.csv", index_col=0)
df_4 = df_4.drop(columns=['count', 'hate_speech', 'offensive_language', 'neither'], axis=1)
df_4.columns = ['target', 'text']
df_4 = df_4.replace({'target': {1: 0, 2: 0, 0: 1}})
preprocess_tweets(df_4, 'text')
df_4 = df_4[['target', 'text']]
df_4.head()

# display value counts for target variable
df_4.target.value_counts()

# concatenate dataframes together for combined dataframe 
df_combined = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True)
df_combined


# get new value counts for target variable
df_combined.target.value_counts()

# separate into hate and non-hate datasets
hate2 = df_combined[df_combined.target == 1]
non_hate2 = df_combined[df_combined.target == 0]

# separate into features and target for train_test_split
X_h2 = hate2.text
y_h2 = hate2.target
X_nh2 = non_hate2.text
y_nh2 = non_hate2.target

# perform 75-15-10 split on dataset
X_h_tr2, X_h_val2, y_h_tr2, y_h_val2 = train_test_split(X_h2, y_h2, test_size=0.25, random_state=42)
X_h_val2, X_h_tt2, y_h_val2, y_h_tt2 = train_test_split(X_h_val2, y_h_val2, test_size=0.4, random_state=42)
X_nh_tr2, X_nh_val2, y_nh_tr2, y_nh_val2 = train_test_split(X_nh2, y_nh2, test_size=0.25, random_state=42)
X_nh_val2, X_nh_tt2, y_nh_val2, y_nh_tt2 = train_test_split(X_nh_val2, y_nh_val2, test_size=0.4, random_state=42)

# concatenate to reform training dataset
X_tr2 = pd.concat((X_h_tr2, X_nh_tr2), ignore_index=True)
y_tr2 = pd.concat((y_h_tr2, y_nh_tr2), ignore_index=True)
train2 = pd.concat([X_tr2, y_tr2], axis=1)

# remove brackets on list to create list of strings
train2.text = train2.text.apply(lambda x: str(x)[1:-1]) 
train2.head()

# concatenate to reform validation dataset
X_val2 = pd.concat((X_h_val2, X_nh_val2), ignore_index=True)
y_val2 = pd.concat((y_h_val2, y_nh_val2), ignore_index=True)
val2 = pd.concat([X_val2, y_val2], axis=1)
val2.text = val2.text.apply(lambda x: str(x)[1:-1]) 
val2.head()

# separate into feature and target variables
X_tr2 = train2.text
X_val2 = val2.text
y_tr2 = train2.target
y_val2 = val2.target

# vectorize data for training
vec = TfidfVectorizer()
tfidf_tr2 = vec.fit_transform(X_tr2)
tfidf_val2 = vec.transform(X_val2)

dum_clf = DummyClassifier(strategy='most_frequent').fit(tfidf_tr2, y_tr2)
y_pr_clf = dum_clf.predict(tfidf_val2)
get_metrics(tfidf_val2, y_val2, y_pr_clf, dum_clf)

nb2 = MultinomialNB().fit(tfidf_tr2, y_tr2)
y_pr_nb_val2 = nb2.predict(tfidf_val2)
get_metrics_confusion(tfidf_val2, y_val2, y_pr_nb_val2, nb2)

rf2 = RandomForestClassifier(n_estimators=100, random_state=42).fit(tfidf_tr2, y_tr2)
y_pr_rf_val2 = rf2.predict(tfidf_val2)
get_metrics_confusion(tfidf_val2, y_val2, y_pr_rf_val2, rf2)

log2 = LogisticRegression(random_state=42).fit(tfidf_tr2, y_tr2)
y_pr_log_val2 = log2.predict(tfidf_val2)
get_metrics_confusion(tfidf_val2, y_val2, y_pr_log_val2, log2)

svc2 = svm.LinearSVC(random_state=42).fit(tfidf_tr2, y_tr2)
y_pr_svc_val2 = svc2.predict(tfidf_val2)
get_metrics_2(tfidf_val2, y_val2, y_pr_svc_val2, svc2)

abc2 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    random_state=42
    ).fit(tfidf_tr2, y_tr2)
y_pr_ada_val2 = abc2.predict(tfidf_val2)
get_metrics_confusion(tfidf_val2, y_val2, y_pr_ada_val2, abc2)

gbc2 = GradientBoostingClassifier(random_state=42).fit(tfidf_tr2, y_tr2)
y_pr_gbc_val2 = gbc2.predict(tfidf_val2)
get_metrics_confusion(tfidf_val2, y_val2, y_pr_gbc_val2, gbc2)

tfidf_val2

metrics5 = pd.read_csv("data/metrics/metrics5.csv", index_col=0)
metrics5

train = pickle.load(open("pickle/train.pickle", "rb"))
val = pickle.load(open("pickle/val.pickle", "rb"))
test = pickle.load(open("pickle/test.pickle", "rb"))
train.target.value_counts()

train['label'] = train.target.apply(lambda x: str(x))
val['label'] = val.target.apply(lambda x: str(x))
test['label'] = test.target.apply(lambda x: str(x))
X_train = train.tweet
X_val = val.tweet
X_test = test.tweet
y_train = train.target
y_val = val.target
y_test = test.target

train_tagged = train.apply(lambda x: TaggedDocument(words=x['tweet'], tags=[str(x.label)]), axis=1)
train_tagged.values[30]
val_tagged = val.apply(lambda x: TaggedDocument(words=x['tweet'], tags=[str(x.label)]), axis=1)
val_tagged.values[10]

cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

%%time
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_tr_dbow, X_tr_dbow = vec_for_learning(model_dbow, train_tagged)
y_val_dbow, X_val_dbow = vec_for_learning(model_dbow, val_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5).fit(X_tr_dbow, y_tr_dbow)
y_pred_dbow = logreg.predict(X_val_dbow)

get_metrics_3(X_val_dbow, y_val_dbow, y_pred_dbow, logreg)
get_confusion(y_val_dbow, y_pred_dbow)
cores = multiprocessing.cpu_count()
model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=5, min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

%%time
for epoch in range(30):
    model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dmm.alpha -= 0.002
    model_dmm.min_alpha = model_dmm.alpha

    y_tr_dmm, X_tr_dmm = vec_for_learning(model_dmm, train_tagged)
y_val_dmm, X_val_dmm = vec_for_learning(model_dmm, val_tagged)

logreg_2 = LogisticRegression().fit(X_tr_dmm, y_tr_dmm)
y_pred_dmm = logreg_2.predict(X_val_dmm)

get_metrics_3(X_val_dmm, y_val_dmm, y_pred_dmm, logreg_2)
get_confusion(y_val_dmm, y_pred_dmm)