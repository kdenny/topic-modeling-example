
## Document parsing and preprocessing
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from pprint import pprint
import csv

## The actual topic modeling
import gensim
from gensim import corpora

def get_topics():
    topics = []
    with open("philly_articles.csv",'r') as cfile:
        rd = csv.DictReader(cfile)
        for row in rd:
            ta = row['title'] + '. '
            if str(row['text']).strip() != '':
                ta += row['text'].replace("<a","").replace("href=","").replace("</a>","")

                topics.append(ta)

    return topics

# compile documents
dc = get_topics()
doc_complete = []
c = 0
for cc in dc:
    # if c < 200:
    doc_complete.append(cc)

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
no_words_article = ['philadelphia','philly','year','city','street','one','plan','2017','2016','2015','2014','2013','2012','2011','year','also','project','new','developer','development','building','center','said','say','two','site','space','first','it','street','square','project','home','time','last','design','get','property','lot','next','would','see','around','house','look','land','space']

# stop.add("design")
# stop.add("building")
# stop.add("square")
# stop.add("construction")
# stop.add("philadelphia")
# stop.add("callowhill")
# stop.add("year")
# stop.add("project")
# stop.add("property")
# stop.add("year")
# stop.add("site")
# stop.add("2017")
# stop.add("2016")
# stop.add("2015")
# stop.add("2014")
# stop.add("2013")
# stop.add("2012")
# stop.add("2011")
for wd in no_words_article:
    stop.add(wd)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

print(doc_clean)

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(doc_clean)


# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word = dictionary, passes=100)

tp = ldamodel.print_topics(num_topics=20, num_words=20)
pprint(tp)


def test_new_article(text):
    new_text = clean(text).split()
    print(new_text)
    d = [dictionary.doc2bow(new_text)]
    predictions = ldamodel[d]
    for p in predictions:
        print(p)
    print(predictions)

t = dc[253]
test_new_article(t)