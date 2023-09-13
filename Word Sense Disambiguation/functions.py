# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.corpus import senseval
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate
from nltk.corpus import wordnet
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from nltk.metrics.scores import precision, recall, f_measure, accuracy
from nltk.corpus import wordnet_ic

def load_data():

    #nltk.download('senseval')
    #nltk.download('wordnet')
    #nltk.download('stopwords')
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('wordnet_ic')

    #instantiate BOW data and labels
    data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]
    labels = [inst.senses[0] for inst in senseval.instances('interest.pos')]

    #instantiate a dictionary with collocational features using POS-tags and Ngrams
    data_col = [collocational_features_extend(inst) for inst in senseval.instances('interest.pos')]

    return data, data_col, labels

#Extend collocational features by adding POS-tags and Ngrams
def collocational_features_extend(inst):
    p = inst.position
    return {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0],
        "w-2_pos": 'NULL' if p < 2 else inst.context[p-2][1],
        "w-1_pos": 'NULL' if p < 1 else inst.context[p-1][1],
        "w+1_pos": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][1],
        "w+2_pos": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][1],
        "w-2_gram": 'NULL' if p < 2 else inst.context[p-2][0] + ' ' + inst.context[p-1][0] + ' ' + inst.context[p][0],
        "w-1_gram": 'NULL' if p < 1 else inst.context[p-1][0] + ' ' + inst.context[p][0],
        "w+1_gram": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p][0] + ' ' +  inst.context[p+1][0],
        "w+2_gram": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p][0] + ' ' +  inst.context[p+1][0] + ' ' + inst.context[p+2][0]
    }

#class for Supervised Word Sense Disambiguation
class WSD_Model():
    def __init__(self, labels, collocational_vector=False, concatenate=False):

        self.collocational=collocational_vector
        self.concatenate=concatenate

        #instantiate sk-learn classes    
        self.classifier = MultinomialNB()
        self.labelencoder = LabelEncoder()
        self.stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

        # encoding labels for multi-class
        self.labelencoder.fit(labels)
        self.labels = self.labelencoder.transform(labels)

    def evaluate(self, data, data_col=None):

        #compute BOW vector
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(data)
        
        #compute collocational vector
        dvectorizer = DictVectorizer(sparse=False)
        dvectors = dvectorizer.fit_transform(data_col)

        #compute score for BOW vector
        scores_bow = cross_validate(self.classifier, vectors, self.labels, cv=self.stratified_split, scoring=['f1_micro'])
        score_bow = sum(scores_bow['test_f1_micro'])/len(scores_bow['test_f1_micro'])

        #compute score for collocational features vector
        scores_col = cross_validate(self.classifier, dvectors, self.labels, cv=self.stratified_split, scoring=['f1_micro'])
        score_col = sum(scores_col['test_f1_micro'])/len(scores_col['test_f1_micro'])

        if not self.concatenate:
            if self.collocational:
                print('Collocational features average f1-micro score: {:.3f}'.format(score_col))
            else:
                print('Bow features average f1-micro score: {:.3f}'.format(score_bow))

        if self.concatenate:
            #concatenate BOW vector with collocational feature vector
            concatenate_vectors = np.concatenate((vectors.toarray(), dvectors), axis=1)

            #compute score for the concatenate vector
            concatenate_scores = cross_validate(self.classifier, concatenate_vectors, self.labels, cv=self.stratified_split, scoring=['f1_micro'])
            concatenate_score = sum(concatenate_scores['test_f1_micro'])/len(concatenate_scores['test_f1_micro'])
            print('Bow + Collocational features average f1-micro score: {:.3f}'.format(concatenate_score))

def preprocess(text):
    
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}
    sw_list = stopwords.words('english')# Add the stopword list
    
    lem = WordNetLemmatizer()
    
    # tokenize, if input is text
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    
    # compute pos-tag
    tagged = nltk.pos_tag(tokens, tagset="universal")
    
    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    
    # lemmatize
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    
    # unique the list
    tagged = list(set(tagged))
    
    return tagged

def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)

    # let's get senses for each
    senses = [(w, wordnet.synsets(l, p)) for w, l, p in lemma_tags]

    # let's get their definitions
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            # let's tokenize, lowercase & remove stop words 
            def_list = []
            for s in sense_list:
                defn = s.definition()
                # let's use the same preprocessing
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions

def get_top_sense(words, sense_list):
    
    # get top sense from the list of sense-definition tuples   
    # assumes that words and definitions are preprocessed identically
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    
    return val, sense

def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False):
    
    context_senses = get_sense_definitions(set(context_sentence)-set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    scores = []

    for senses in context_senses:
        for sense in senses[1]:
            # Append the score with the highest score and relative value 
            # Compare sense[1] with synsets
            # get_top_sense might help here
            scores.append(get_top_sense(sense[1], synsets))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        # We remove 0 scores, senses without overlapping
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            #We need to select the most common syn. Counter function might help here
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            # The same as above but using scores instead of filtered_scores
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores) # Get the maximum of scores.
    return best_sense

def get_top_sense_sim(context_sense, sense_list, similarity):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')

    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))    
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))       
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense

def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, 
                    synsets=None, majority=True):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    
    # Here you may have some room for improvement
    # For instance instead of using all the definitions from the context
    # you pick the most common one of each word (i.e. the first)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    
    return best_sense

def pedersen(context_sentence, ambiguous_word, similarity="resnik", pos=None, 
                    synsets=None, threshold=0.1):
    
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))

    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    synsets_scores = {}
    for ss_tup in synsets:
        ss = ss_tup[0]
        if ss not in synsets_scores:
            synsets_scores[ss] = 0
        for senses in context_senses:
            scores = []
            for sense in senses[1]:
                if similarity == "path":
                    try:
                        scores.append((sense[0].path_similarity(ss), ss))
                    except:
                        scores.append((0, ss))    
                elif similarity == "lch":
                    try:
                        scores.append((sense[0].lch_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "wup":
                    try:
                        scores.append((sense[0].wup_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "resnik":
                    try:
                        scores.append((sense[0].res_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "lin":
                    try:
                        scores.append((sense[0].lin_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "jiang":
                    try:
                        scores.append((sense[0].jcn_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                else:
                    print("Similarity metric not found")
                    return None
            value, sense = max(scores)
            if value > threshold:
                synsets_scores[sense] = synsets_scores[sense] + value

    values = list(synsets_scores.values())
    #if sum(values) == 0:
    #    print('Warning all the scores are 0')
    senses = list(synsets_scores.keys())
    best_sense_id = values.index(max(values))
    return senses[best_sense_id]

def evaluate_wsd(metric='lesk_original'):
    
    # Let's create mapping from convenience
    mapping = {
        'interest_1': 'interest.n.01',
        'interest_2': 'interest.n.03',
        'interest_3': 'pastime.n.01',
        'interest_4': 'sake.n.01',
        'interest_5': 'interest.n.05',
        'interest_6': 'interest.n.04',
    }

    refs = {k: set() for k in mapping.values()}
    hyps = {k: set() for k in mapping.values()}
    refs_list = []
    hyps_list = []

    # since WordNet defines more senses, let's restrict predictions
    synsets = []
    for ss in wordnet.synsets('interest', pos='n'):
        if ss.name() in mapping.values():

            defn = ss.definition() # estract the defitions
            tags = preprocess(defn) # Preproccess the definition
            toks = [l for w, l, p in tags] # From tags extract the tokens

            synsets.append((ss,toks))

    for i, inst in enumerate(senseval.instances('interest.pos')):
        txt = [t[0] for t in inst.context]
        raw_ref = inst.senses[0] #get first sense

        # Use original LESK or similarity LESK, for input parameters copy paste from above.
        if(metric == 'lesk_original'):
            hyp = original_lesk(txt, txt[inst.position], synsets=synsets, majority=True).name()
        elif(metric == 'lesk_similarity'):
            #hyp = lesk_similarity(txt, txt[inst.position], synsets=synsets, majority=True).name()
            hyp = pedersen(txt, txt[inst.position], synsets=synsets).name()
        else:
            return
        
        ref = mapping.get(raw_ref)

        # for precision, recall, f-measure        
        refs[ref].add(i)
        hyps[hyp].add(i)

        # for accuracy
        refs_list.append(ref)
        hyps_list.append(hyp)

    print("Acc:", round(accuracy(refs_list, hyps_list), 3))

    for cls in hyps.keys():
        p = precision(refs[cls], hyps[cls])
        r = recall(refs[cls], hyps[cls])
        f = f_measure(refs[cls], hyps[cls], alpha=1)

        print("{:15s}: p={:.3f}; r={:.3f}; f={:.3f}; s={}".format(cls, p, r, f, len(refs[cls])))