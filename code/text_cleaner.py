import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    """
    TextCleaner class for cleaning text data using various preprocessing 
    steps.

    Usage:
    >> cleaner = TextCleaner()
    >> cleaned_text = cleaner.clean_text(text_input)
    >> df['text'] = df.apply(lambda r: cleaner.clean_text(r.text), axis=1)
    
    """
    def __init__(self):
        """
        Initialize the TextCleaner class.

        Downloads NLTK resources if not already downloaded and 
        initializes lemmatizer and stop words.
        
        """
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')).union({'goog'})

    def clean_text(self, text, remove_digits=True, remove_special_chars=True):
        """
        Clean text in the dataset.

        :param text: text that is present in the input file.
        :type  text: str
        :param remove_digits: flag to remove digits from the text.
        :type  remove_digits: bool
        :param remove_special_chars: flag to remove special characters 
                                     from the text.
        :type  remove_special_chars: bool
        :return: string that has been cleaned using Wordnet Lemmatizer, etc.
        :rtype: str
        
        """
        text = str(text).lower()  # convert to lowercase
        text = re.sub(r'^https?:\/\/.*[\s]*', '', text)  # remove links
        words = word_tokenize(text)
        
        final_tokens = []
        for w in words:
            # remove punctuation
            w = "".join(["" if c in string.punctuation else c for c in w])
            if w != "" and w not in self.stop_words:  # process only non-stopwords
                if remove_digits:
                    w = re.sub(r'\d', '', w)  # remove digits
                if remove_special_chars:
                    # remove special characters
                    w = re.sub(r'[^a-zA-Z0-9\s]', '', w)
                # using WordNet for lemmatizing
                final_tokens.append(self.lemmatizer.lemmatize(w))
        
        cleaned_txt = " ".join(final_tokens)
        return cleaned_txt
