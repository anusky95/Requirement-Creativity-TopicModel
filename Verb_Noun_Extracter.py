import re
import sys
import nltk


def POSTagger(file_name):
    with open(file_name,'r') as f:
        with open('POSTaggedFile.txt','w')as dest:
            for line in f:
                tokens = nltk.word_tokenize(line)
                tags = nltk.pos_tag(tokens)
                #nounVerb = [word for word, pos in tags if (pos == 'NNS' or pos == 'VB' or pos == 'NN' or pos == 'JJ' or pos == 'TO' or pos == 'CC' or pos == 'IN' or pos == 'DT' == pos == 'VBP')]
                nounVerb = [word for word, pos in tags if (pos == 'NNS' or pos == 'VB' or pos == 'NN')]
                modNounVerb = ' '.join(nounVerb)
                formattedResult = '{0}{1}{2}'.format('"', modNounVerb.rstrip('\n'), '"')
                #Print to the output console
               # print('{0}{1}{2}'.format('"',modNounVerb.rstrip('\n'),'"'))
                print(formattedResult,file=dest)

def main():
    file_name = sys.argv[1]
    POSTagger(file_name)

if __name__ == "__main__":
    main()