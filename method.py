import nltk
nltk.download('punkt')
import collections
import re
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import sumy

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.kl import KLSummarizer
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer 
from sumy.summarizers.lex_rank import LexRankSummarizer

# source:
# http://www.di.ubi.pt/~jpaulo/competence/general/(1958)Luhn.pdf
# https://github.com/ptwobrussell/Mining-the-Social-Web
# https://github.com/miso-belica/sumy
# https://pythonfix.com/code/sumy-code-examples/
# https://www.machinelearningplus.com/nlp/text-summarization-approaches-nlp-example/
# https://iq.opengenus.org/k-l-sum-algorithm-for-text-summarization/
# https://www.youtube.com/watch?v=8p9nHmtwk0o

class Method:
    def __init__(self, sentence_count, file=None, text=None, url=None):
        self.sentence_count = sentence_count
        self.text = text
        self.file = file
        self.url = url
    
    def klsum(self):
        language = "english"

        # or for plain text files
        if self.url is not None:
            parser = HtmlParser.from_url(self.url, Tokenizer(language))
        elif self.file is not None:
            parser = PlaintextParser.from_file(self.file, Tokenizer(language))
        elif self.text is not None:
            parser = PlaintextParser.from_string(self.text, Tokenizer(language))
            
        stemmer = Stemmer(language)
        summarizer = KLSummarizer()
        summarizer.stop_words = get_stop_words(language)

        summary = summarizer(parser.document,self.sentence_count)
        
        summarize_text = ""
        for sentence in summary:
            summarize_text += str(sentence)
        
        return summarize_text

        
    def luhn(self):
        sentences_score = []
        summary_count = self.sentence_count
        if self.file:
            with open(self.file, 'r') as file:
                txt = file.read().replace('\n','')
        if self.text:
            txt = self.text.replace('\n','')
        if self.url:
            req = Request(self.url)
            html_page = urlopen(req)
            soup = BeautifulSoup(html_page, "html.parser")
            html_text = soup.get_text()
            txt = html_text.replace('\n','')
        sentences = [s for s in nltk.tokenize.sent_tokenize(txt)]
        normalized_sen = [s.lower() for s in sentences]
        normalized_txt = re.compile(r'[^0-9^a-z^A-Z\s]').sub('',txt)
        words = [w.lower() for w in nltk.tokenize.word_tokenize(normalized_txt)]
        freq_word = collections.Counter(words)
        freq_dict = dict(freq_word)
        keywords = set()
        for word in freq_dict:
            word_percentage = freq_dict[word] * 1.0/len(words)
            if word_percentage <= 0.5 and word_percentage >= 0.001:
                keywords.add(word)
        for s in sentences:
            sentences_score.append((self.sentence_score(s,keywords),s))
        sentences_score.sort(reverse = True)
        result = []
        result_num = min(summary_count, len(normalized_sen))
        for i in range(result_num):
            result.append(sentences_score[i][1])
        return ''.join(result)

    def sentence_score(self, sentence, keyword):
        words = [w.lower() for w in nltk.tokenize.word_tokenize(sentence)]
        start = 0
        end = -1
        for i in range(len(words)):
            if words[i] in keyword:
                start = i
                break
        for j in range(len(words) - 1, 0 , -1):
            if words[i] in keyword:
                end = j
                break
        if start > end:
            return 0
        size = end - start + 1
        freq = 0
        for each_word in words:
            if each_word in keyword:
                freq += 1
        return freq * freq * 1.0 / size
    
    def lsa(self):
        if self.url is not None:
            parser = HtmlParser.from_url(self.url, Tokenizer("english"))
        elif self.file is not None:
            parser = PlaintextParser.from_file(self.file, Tokenizer("english"))
        elif self.text is not None:
            parser = PlaintextParser.from_string(self.text, Tokenizer("english"))
        
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document,self.sentence_count)
        
        summarize_text = ""
        for sentence in summary:
            summarize_text += str(sentence)
        
        return summarize_text
    
    def textRank(self):
        if self.url is not None:
            parser = HtmlParser.from_url(self.url, Tokenizer("english"))
        elif self.file is not None:
            parser = PlaintextParser.from_file(self.file, Tokenizer("english"))
        elif self.text is not None:
            parser = PlaintextParser.from_string(self.text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, self.sentence_count)

        summarize_text = ""
        for sentence in summary:
            summarize_text += str(sentence)
        
        return summarize_text


