# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.parse.dependencygraph import DependencyGraph
from spacy.tokenizer import Tokenizer
import spacy 
import spacy_conll
from nltk.parse import DependencyEvaluator
import stanza
import spacy_stanza
from nltk.corpus import dependency_treebank
import nltk


#defining an utility function
def from_token_to_sent(token_list: list) -> str:
    sent = token_list[0]
    for i in range(1,len(token_list)):
        sent += ' ' + token_list[i]
    return sent

#collect data phrases
def load_data(n_sentences=100):

    # downloading treebank
    #nltk.download('dependency_treebank')

    train_sents = dependency_treebank.sents()[-n_sentences:]
    test_sents = dependency_treebank.parsed_sents()[-n_sentences:]

    return train_sents, test_sents


class DependencyParser():
    def __init__(self, version='spacy'):

        self.version=version

        if version=='spacy':
            # Load the spacy model
            self.nlp = spacy.load("en_core_web_sm")

            # Set up the conll formatter 
            self.config = {"ext_names": {"conll_pd": "pandas"},
                    "conversion_maps": {"DEPREL": {"nsubj": "subj"}}}
            
        elif version=='stanza':
            #tokenize_pretokenized used to tokenize by white space 
            self.nlp = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)

            # Set up the conll formatter
            self.config = {"ext_names": {"conll_pd": "pandas"},
                    "conversion_maps": {"DEPREL": {"nsubj": "subj", "root":"ROOT"}}}


        # Add the formatter to the pipeline
        self.nlp.add_pipe("conll_formatter", config=self.config, last=True)

        if version=='spacy':
            # Split by white space
            self.nlp.tokenizer = Tokenizer(self.nlp.vocab) 

    def parse(self, train_sents):
        d_graph = []

        for tokens in train_sents:
            
            #parse each sentence with spaCy parser
            doc = self.nlp(from_token_to_sent(tokens))

            # Convert to a pandas object
            df = doc._.pandas

            # Select the columns accoroding to Malt-Tab format
            tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)

            # Get the DepencecyGraph
            dp = DependencyGraph(tmp)
            
            #store each DependencyGraph to evaluate later
            d_graph.append(dp)

        #example of DependencyGraph
        print(f'#### Example of Dependency Graph ({self.version} Version)')
        d_graph[0].tree().pretty_print()

        return d_graph

def evaluate(d_graph_spacy, d_graph_stanza, test_sents):
    spacy_de = DependencyEvaluator(d_graph_spacy, test_sents)
    stanza_de = DependencyEvaluator(d_graph_stanza, test_sents)
        
    spacy_las, spacy_uas = spacy_de.eval()
    stanza_las, stanza_uas = stanza_de.eval()

    # no labels, thus identical
    print("SpaCy Evaluation")
    print(f"LAS: {spacy_las} \nUAS: {spacy_uas}")
    print("Stanza Evaluation")
    print(f"LAS: {stanza_las} \nUAS: {stanza_uas}")