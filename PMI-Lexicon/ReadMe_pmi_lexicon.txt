Generation Methodology:
This lexicon was generated based on dialogue dataset which consists of two TV series in addition to Cornell Movie Dialogue Corpus. Out of a vocabulary of 4,209,560 distinct words, we keep all words for which there exists at least one representative words for at least one emotion in the GALC Lexicon within a context window of a specified length 10. This means if a word will not be contained in the PMI lexicon if it is too far from any emotional word. This lexicon comprises only unigrams. 
Pre-processing: 
- Lower-case conversion
- Filtering out stopwords
- removal of punctuation
- Lemmatization  
Format: 
Each line corresponds to the emotional vector of a word in the dataset. Each entry in the vector is the semantic relatedness score: the arithmetic mean of the semantic similarity of the word with respect to each representative word in the set of the representative words for a specific emotion. The ordered list of emotions is found in file GEW_20emotions.txt. For example, the word superior has the following vector: 
superior	0.09092505438200678	0	0	0	0	0	0	0	0	0	0	0	0.08260085595429488	0	0	0	0	0	0	0.23933122172913507	
its involvment score is 0.09092505438200678


  

