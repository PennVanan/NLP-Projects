import sys
import nltk
import numpy
import os

from nltk import *

alasar_mi_dict=dict()
alasar_vals_mi = list()
alasar_keys_mi = list()


def get_all_files(directory):
       """
       Function: get_all_files
       -----------------------
       list files of any depth in a directory
       directory: complete directory path
       returns: a list of relative file paths
       """
       L1 = list();
       L2 = list();
       l = len(directory)
       if os.path.isdir(directory) == False:
         print "Error: Enter a valid directory path"
         return       
       print "Extracting files..."
       #Marking directories and files in Level-1
       for dirs in os.listdir(directory):
         if os.path.isdir(os.path.join(directory,dirs)) == True:
               L1.append(os.path.join(directory, dirs));
               L2.append(1);
         else:
               L1.append(os.path.join(directory, dirs));
               L2.append(2);
       #Iteratively marking all directories and files
       while True:
           j=0;
           for i in xrange(0, len(L1)):
               if(L2[i]==1):  j=j+1
               if L2[i]==1:
                  L2[i] = 0;
                  for dirs in os.listdir(L1[i]):
                     if os.path.isdir(os.path.join(L1[i],dirs)) == True:
                        L1.append(os.path.join(L1[i],dirs));
                        L2.append(1);
                     else:
                        L1.append(os.path.join(L1[i], dirs));
                        L2.append(2);
           if(j==0):  break;
       L3=list()
       #Adding all files that have been discovered
       for i in xrange(0, len(L1)):
           if(L2[i]==2):  L3.append(L1[i]);
       #Extracting the relative file paths
       i=0
       for i in range(0, len(L3)):
           a = L3[i]
           if(directory[l-1] == '/'):
                 a=a[l:]
           else:
                 a = a[l+1:]
           L3[i] = a
       for x in L3:
              print x + "\n"
       return L3;


def load_file_tokens(filepath):
       """
       Function: load_file_tokens
       --------------------------
       splits the file contents to lowercased alphanumeric tokens
       filepath: absolute path of the file
       returns: a list of tokens converted to lowercase
       """
       tklist = list()
       f=open(filepath,'r');
       text=f.read()
       sentences=sent_tokenize(text)
       words=list()
       for i in xrange(0,len(sentences)):
          t=word_tokenize(sentences[i])
          #Discard fullstops
          for a in t:
             if a is '.':  
               continue
             else:
               words.append(a)
       tokens=list()
       #Extract tokens
       for i, tkn in enumerate(words):
         wordList = re.sub("[^\w]"," ", tkn).split()
         if len(wordList)>0:
           for x in wordList:
             tokens.append(x.lower())
       f.close();
       return tokens;


def load_collection_tokens(directory):
       """
       Function: load_collection_tokens
       --------------------------------
       directory: absolute path of the directory
       returns: a list of tokens collected from all files
       """
       inp = list()
       tkl = list()
       inp = get_all_files(directory)
       for filename in inp:
            filename = directory + "/" + filename
            print "Processing file: " + filename
            tmp = load_file_tokens(filename);
            if(len(tmp)>0):
               for i in xrange(0, len(tmp)):
                  tkl.append(tmp[i]);
       return tkl;


def get_tf(l):
       """
       Function: get_tf
       ---------------------------------------------
       l: 1D list containing all terms from all docs
       returns: dictionary with -
                unique term as key 
                normalized term frequency as value
                TF(w) = (count(w)/most_frequent_word_count)
       """
       dictionary = dict()
       print "Gathering unique tokens..." 
       for x in range(0, len(l)):
         if(l[x] in dictionary):
                dictionary[l[x]] = dictionary[l[x]] + 1
         else:
                a=l[x]
                c=['\'', '?',',',')','(','[',']']
                if(l[x] not in c and a[0]!='\''):
                    dictionary[l[x]] =  1
       dictionary['UNK'] = 1
       print dictionary
       val=list(dictionary.values())
       kys=list(dictionary.keys())
       max_val = max(val)
       print "Length of dictionary"
       print len(dictionary)
       dictionary=[(k,v) for v,k in sorted([(v,k) for k,v in dictionary.items()],reverse=True)]
       i=1
       d=dict() 
       print "Calculating TF"
       for k,v in dictionary:
          print k + " occurs " + str(v) + " times\n"
          tv=float(v)/float(max_val);  #Normalized TF
          d[k]=tv
          print d[k]
          print " \n"
       i=0
       val=d.values()
       keys=d.keys()
       vals_tf,keys_tf = quick_sort_recursive(val,keys, 0, len(val)-1)
       tf_dict = dict()
       for i in xrange(0,len(keys_tf)-1):
           tf_dict[keys_tf[i]] = vals_tf[i]
           i = i+1
       return tf_dict

def gen2dlist(dirpath):
       """
       Function: gen2dlist
       ---------------------------------------------
       dirpath: directory containing the documents
       returns: 2D list with a lists of words grouped 
                in the order of list of documents
       """
       elements = []
       inp = get_all_files(dirpath)
       i=0
       j=0
       for filename in inp:
             filename = dirpath + "//" + filename
             print "Processing" + filename
             elements.append([])  #New document
             tmp = load_file_tokens(filename);
             if(len(tmp)>0):
               for j in xrange(0, len(tmp)):
                  elements[i].append(tmp[j]); #List of words in the document
             i=i+1;
       print elements[0][0]
       return elements;


def get_idf(itemlist):
       """
       Function: get_idf
       ---------------------------------------------------
       itemlist: 2D list containing all terms from each doc
       returns: dictionary with -
                unique term as key
                corresponding IDF value
            IDF(w) = ln(N/DF(w))
       """
       dictionary = dict()
       doclen = len(itemlist)
       i=0
       j=0
       #Extracting unique tokens
       for doc in itemlist:
          print "In document " + str(j) + " of " + str(doclen) + "\n"
          j=j+1
          for word in doc:
            i=i+1
            if(word in dictionary):
                dictionary[word] = dictionary[word] + 1
            else:
                a=word
                c=['\'', '?',',',')','(','[',']']
                if(a not in c and a[0]!='\''):
                    dictionary[word] =  1
       df = dict()
       wlen = len(dictionary)
       i=0
       #Counting the occurence of each token in all documents
       for word in dictionary:
          i=i+1
          print "Examining word " + str(i) + " of " + str(wlen)
          cnt=0; doc_len=0;
          for doc in itemlist:
             doc_len=doc_len+1
             if word in doc:
               cnt=cnt+1
             df[word] = cnt;
       df['UNK'] = 1
       keys=df.keys()
       val=df.values()
       idf = dict()
       for word in df:
           idf[word] = numpy.log(float(doc_len)/float(df[word])) #Calculating IDF
       print idf
       keys=idf.keys()
       val=idf.values()
       return idf

def get_tfidf_top(dict1, dict2, k):
       """
       Function: get_tfidf_top
       -------------------------------------------------------------------
       dict1 - TF dictionary
       dict2 - IDF dictionary
       returns: Top 'k' terms of TF * IDF values sorted in descending order
              TF_IDF(w) = TF(w) * IDF(w)
       """
       tf=dict()
       idf=dict()
       tfidf = dict()
       print len(dict1)
       print len(dict2)
       #Sort TF
       for key in sorted(dict1):
         tf[key] = dict1[key]
       #Sort IDF
       for key in sorted(dict2):
         idf[key] = dict2[key]
       #Calculate TF-IDF
       for x in tf:
         tfidf[x] = tf[x]*idf[x]
       result = list()
       i=0
       tempdict=tfidf.copy()
       #Sort TF-IDF in decreasing order of values
       while True:
         val=tempdict.values()
         keys=tempdict.keys()
         tmp=keys[val.index(max(val))]
         result.append(tmp);
         del tempdict[tmp]
         i=i+1
         if tempdict:
           continue
         else:
            break
       #Return top 'k' terms
       finalres = list()
       for i in xrange(0,k-1):
          finalres.append(result[i])
       return finalres


def partition(val,key, start, end):
       """
       Function: partition
       ---------------------------------------------------
       val, key - values and keys to be partitioned
       start, end - limits of the list
       """
       pos = start
       for i in range(start, end):
          if val[i] < val[end]:
            val[i],val[pos] = val[pos],val[i]
            key[i],key[pos] = key[pos],key[i]
            pos += 1
       val[pos],val[end] = val[end],val[pos]
       key[pos],key[end] = key[end],key[pos]
       return pos

def quick_sort_recursive(val, key, start, end):
       """
       Function: quick_sort_recursive
       ---------------------------------------------------------
       val, key - values and keys to be sorted
       start, end - limits of the list
       returns: keys and values sorted by values in reverse order
       """
       sys.setrecursionlimit(60000)
       if start < end:
         pos = partition(val, key, start, end)
         quick_sort_recursive(val, key, start, pos - 1)
         quick_sort_recursive(val, key, pos+1, end)
       return val[::-1], key[::-1]  #Sorting in reverse order


def get_mi_top(bg_terms, topic_terms, k):
       """
       Function: get_mi_top
       -------------------------------------------------------------------
       bg_terms: Terms in all documents
       topic_terms: Terms specific to the target topic
       k: top 'k' MI values to be returned
       returns: a list of 'k' sorted terms with MI values in decreasing order
            MI(topic_i, word_w) = ln(P(word_w|topic_i)/p(word_w))
       """
       s = set(topic_terms)
       s = sorted(s)
       topic = list(s)
       total_count_company = len(topic_terms)
       total_count_all = len(bg_terms)
       c=['\'','\"','%', '?',',',')','(','[',']',';',':']
       for t in topic:
         if t not in c and t[0]!='\\':  
            term_count_company = topic_terms.count(t)
            term_count_all = bg_terms.count(t)
            prob_w_by_topic_t = float(term_count_company)/float(total_count_company)
            prob_w = float(term_count_all)/float(total_count_all)
            if(prob_w) == 0 or term_count_all < 5:
              alasar_mi_dict[t] = 0
            else:
              alasar_mi_dict[t] = float(prob_w_by_topic_t)/float(prob_w)
       tempdict=alasar_mi_dict.copy()
       val=tempdict.values()
       keys=tempdict.keys()
       global alasar_vals_mi
       global alasar_keys_mi
       alasar_vals_mi,alasar_keys_mi = quick_sort_recursive(val,keys, 0, len(val)-1)
       res=list()
       i=0
       for key in alasar_keys_mi:   #Return top 'k' MI values in decreasing order
         res.append(key)
         i=i+1
         if(i==k):
           break
       return res

def write_mi_weights(directory, outfilename):
       """
       Function: write_mi_weights
       -------------------------------------------------------------------
       directory: output directory name
       outfilename: output filename
       returns:  <word MI-value> sequences printed into the target file
       """
       if os.path.exists(directory) == False:
          print "Directory does not exist"
          return -1
       filename = directory + "/" + outfilename
       fo = open(filename,'w')
       for i in xrange(0,len(alasar_keys_mi)):
          if(alasar_vals_mi[i] > 0):
            fo.write(alasar_keys_mi[i] + '\t' + str(alasar_vals_mi[i]) + '\n')
       fo.close();


def get_precision(L1, L2):
       """
       Function: get_precision
       -------------------------------------------------------------------
       L1: top k terms returned by MI/normalized-TF
       L2: top k terms returned by TF-IDF
       returns: precision value 
       """
       intersection = list(set(L1) & set(L2))
       L1NL2 = len(intersection)
       NL1 = len(L1)
       NL2 = len(L2)
       Precision = float(L1NL2)/float(NL1)
       return Precision

def get_recall(L1, L2):
       """
       Function: get_recall
       -------------------------------------------------------------------
       L1: top k terms returned by MI/normalized-TF
       L2: top k terms returned by TF-IDF
       returns: recall value
       """
       intersection = list(set(L1) & set(L2))
       L1NL2 = len(intersection) 
       NL1 = len(L1)
       NL2 = len(L2)
       Recall = float(L1NL2)/float(NL2)
       return Recall

def get_fmeasure(L1, L2):
       """
       Function: get_fmeasure
       -------------------------------------------------------------------
       L1: top k terms returned by MI/normalized-TF
       L2: top k terms returned by TF-IDF
       returns: F-Measure value
       """
       Precision = get_precision(L1,L2)
       Recall = get_recall(L1,L2)
       num = 2.0 * Precision * Recall
       den = Precision + Recall
       fMeasure = num/den
       return fMeasure

def read_file(filename,k):
       """
       Function: read_file
       -------------------------------------------------------------------
       filename: file to be parsed
       k: top 'k' number of tokens to be retrieved from the file which has
          tokens and weights
       returns: top 'k' tokens
       """
       f=open(filename,'r')
       lines = f.readlines()
       f.close()
       l=list()
       i=0
       for lind in xrange(0, len(lines)):
             tokens = word_tokenize(lines[lind])
             word1=tokens[0]  #Isolate token from each line
             l.append(word1)
             i=i+1
             if(i==k):
                break
       return l

def read_brown_cluster():
       """
       Function: read_brown_cluster
       -------------------------------------------------------------------
       returns: a dictionary with words and their corresponding clusterids
       """
       f = open('/home1/c/cis530/hw1/data/brownwc.txt')
       lines = f.readlines()
       f.close()
       brown_dictionary = dict()
       print "Reading Brown Clusters from brownwc.txt..."
       for lind in xrange(0, len(lines)):
             tokens = word_tokenize(lines[lind])
             word1=tokens[1]
             id1=tokens[0]
             brown_dictionary[word1]=id1
       return brown_dictionary


def load_file_clusters(filepath, bc_dict):
       """
       Function: load_file_clusters
       -------------------------------------------------------------------
       filepath - absolute path of the file
       bc_dict - brown cluster dictionary 
       returns: a list of cluster ids corresponding to words in the file
       """
       tokens = load_file_tokens(filepath)
       cluster = list()
       for word in tokens:
           if word in bc_dict:  
                cluster.append(bc_dict[word])
       return cluster

def load_collection_clusters(directory, bc_dict):
       """
       Function: load_file_clusters
       --------------------------------------------------------------------------------
       directory - absolute path of the directory 
       bc_dict - brown cluster dictionary
       returns: a list of cluster ids corresponding to words in all files of directory
       """
       tokens = load_collection_tokens(directory)
       cluster = list()
       for word in tokens:
           a=word
           if a in bc_dict and word: 
                print "Adding word " + word
                cluster.append(bc_dict[word])
       return cluster


def get_idf_clusters(bc_dict):
       """
       Function: get_idf_clusters
       --------------------------------------------------------------------------------
       bc_dict - brown cluster dictionary
       returns: dictionary of IDF values for each cluster id under ?
       """
       global idf
       keylist=bc_dict.keys();
       itemlist = gen2dlist('/home1/c/cis530/hw1/data/all_data/')
       dictionary = dict()
       i=0
       words = [word for doc in itemlist for word in doc]
       print "Words: "
       print words
       s = set(words)
       print "Length: "
       print len(s)
       wordlist = list()
       i=0
       for a in s:    #Collecting unique words
         print i
         if a in bc_dict:
           wordlist.append(a)
         i=i+1
       df = dict()
       i=0;  dummy=0;
       wlen=len(wordlist)
       for word in wordlist:
         i=i+1;
         print "Examining word " + str(i) + " of " + str(wlen)  
         cnt=0; doc_len=0;
         for doc in itemlist:
           doc_len=doc_len+1
           if word in doc:
             cnt=cnt+1
           df[bc_dict[word]] = cnt;  #Updating cluster-id count 
       df['UNK'] = 1
       keys=df.keys()
       val=df.values()
       idf = dict()
       for word in df:
         idf[word] = numpy.log(float(doc_len)/float(df[word]))
       print idf
       keys=idf.keys()
       val=idf.values()
       return idf

def write_tfidf_weights(directory, outfilename):
       """
       Function: write_tfidf_weights
       --------------------------------------------------------------------------------
       directory: output directory name
       outfilename: output filename
       returns: writes <cluster-id TF-IDF-value into each file
       """
       if os.path.exists(directory) == False:
         print "Directory" + directory +" does not exist"
         return -1
       filename = directory + "/" + outfilename
       idf_cl = get_idf_clusters(bc_dict)
       tempdict=idf_cl.copy()
       global values_tfidf
       global keys_tfidf
       values_tfidf=tempdict.values()
       keys_tfidf=tempdict.keys()
       values,keys = quick_sort_recursive(values_tfidf,keys_tfidf, 0, len(values_tfidf)-1)
       res=dict()
       i=0
       k=len(keys)
       for key in keys:
         res[key] = values[i]
         i=i+1
         if(i==k):
           break
       fo = open(filename,'w')
       for k in res:
         fo.write(k + "\t" + str(res[k]) + "\n")
       fo.close();
       print filename


def create_feature_space(list1):
       """
       Function: create_feature_space
       --------------------------------------------------------------------------------
       list1: list of cluster ids
       returns: a dictionary of each cluster id mapped onto an integer
       """
       mapping = dict()
       i=0
       for word in list1:
         mapping[word] = i  #mapping each word to an integer
         i=i+1
       return mapping


def get_dict_val(clusterid,bce_dict):
       """
       Function: get_dict_val
       --------------------------------------------------------------------------------
       clusterid - clusterid whose key needs to be found
       bce_dict - Brown cluster dictionary
       """
       l = len(bce_dict)
       for x in bce_dict.keys():
         if(bce_dict[x]==clusterid):
           return x
       return 'NULL'


def vectorize(feature_space, lst):
       """
       Function: vectorize
       --------------------------------------------------------------------------------
       feature_space - a dictionary of clusterids mapped to integers
       lst - a list of clusterids to be vectorized
       returns: a list of 0s and 1s
       """
       vector = list()
       vallist = feature_space.values();
       for i in vallist:
         word=get_dict_val(i, feature_space)  #search word in featurelist
         if word in lst:
           vector.append('1')
         else:
           vector.append('0')
       return vector

def cosine_similarity(X,Y):
       """
       Function: cosine_similarity
       --------------------------------------------------------------------------------
       X,Y - lists for which cosine similarity has to be calculated
       returns: cosine similarity value
       """
       cosine_sim = 0.000
       N=len(X)
       len2=len(Y)
       numer=0.000; denom1=0.000; denom2=0.000;
       if(N!=len2):
           print 'Error: vectors must be of same length for cosine similarity'; 
           return;
       for i in xrange(0,N):
           numer = numer + (float(X[i]) * (float(Y[i])))
       for i in xrange(0,N):
           denom1 = denom1 + (float(X[i]) * (float(X[i])))
       for i in xrange(0,N):
           denom2 = denom2 + (float(Y[i]) * (float(Y[i])))
       if(numer==0 or denom1==0 or denom2==0):  return 0
       cosine_sim = (numer)/(denom1*denom2);
       return cosine_sim

def rank_doc_sim(rep_file, method, test_path, bc_dict):
       """
       Function: rank_doc_sim
       --------------------------------------------------------------------------------
       rep_file: Representation file containing term weights
       method: method used tf-idf/mi
       test_path: absolute path of the test folder
       bc_dict: Brown cluster dictionary
       returns: a list of (document_name,similarity_value) tuples
       """
       if method=='mi':
        featurelist = list()
        mydict = dict()
        f = open(rep_file,'r');
        i=0
        u=list()
        for line in f:
           t=word_tokenize(line)
           featurelist.append(t[0])
           mydict[t[0]] = t[1]
           i=i+1
        fSpace = create_feature_space(featurelist)
        for a in fSpace:
           u.append(mydict[a])
        i=0
        inp = get_all_files(test_path)
        i=0
        fl=[]
        print "Collecting tokens..."
        for filename in inp:
          filename = test_path + "/" + filename
          fl.append([])
          vtokens = load_file_tokens(filename)
          fl[i].append(vtokens)
          i=i+1
        k=0
        v=[]

        print "Vectorizing tokens..."
        for j in range(0,i):
          print "Vectorizing document " + str(j) + " of " + str(i)
          v.append([])
          v[j].append(vectorize(fSpace, fl[j][0]))
        l=0
        cosine_val = dict()
        for filename in inp:
           print "Processing " + filename + "..."
           cosine_val[filename] = cosine_similarity(u,v[l][0])
           l=l+1
        val=cosine_val.values()
        keys=cosine_val.keys()
        alasar_vals_mi,alasar_keys_mi = quick_sort_recursive(val,keys, 0, len(val)-1)
        res=list()
        i=0
        for key in alasar_keys_mi:
          j=i+1
          res.append(key + "," + str(alasar_vals_mi[i]))
          print str(i) + key + "\n"
          i=i+1
        return res
       elif method== 'tfidf':
        featurelist = list()
        mydict = dict()
        f = open(rep_file,'r');
        i = 0
        u = list()
        for line in f:
           t=word_tokenize(line)
           featurelist.append(t[0])
           mydict[t[0]] = float(t[1])
           i=i+1
           fSpace = create_feature_space(featurelist)
        for a in fSpace:
           u.append(mydict[a])
        inp = get_all_files(test_path)
        i=0
        fl=[]
        print "Collecting tokens..."
        for filename in inp:
           filename = test_path + "/" + filename
           fl.append([])
           vtokens = load_file_clusters(filename,bc_dict)
           fl[i].append(vtokens)
           i=i+1
        k=0
        v=[]
        print "Vectorizing tokens..."
        for j in range(0,i):
          print "Vectorizing document " + str(j) + " of " + str(i) + "..."
          v.append([])
          v[j].append(vectorize(fSpace, fl[j][0]))
        l=0
        cosine_val = dict()
        for filename in inp:
           print "Processing.. " + filename
           cosine_val[filename] = cosine_similarity(u,v[l][0])
           l=l+1
        val=cosine_val.values()
        keys=cosine_val.keys()
        alasar_vals_mi,alasar_keys_mi = quick_sort_recursive(val,keys, 0, len(val)-1)
        res=list()
        i=0
        for key in alasar_keys_mi:
           j=i+1
           res.append(key + "," + str(alasar_vals_mi[i]))
           i=i+1
        return res
       else:
        print "Error: Enter method as mi or tfidf"


def related_file_count(filelist, companyname, directory):
       """
       Function: related_file_count
       --------------------------------------------------------------------------------
       filelist: relative paths of list of retrieved files
       companyname: companyname for which count is performed
       directory: reference directory
       returns: number of files that pertain to companyname
       """
       i=0
       for name in filelist:
         name = directory + "/" + name
       if name.find(companyname)!=-1:
         i=i+1
       return i

def related_file_count_result_file(filename, companyname):
       """
       Function: related_file_count
       --------------------------------------------------------------------------------
       filename: result filename on which count is performed
       companyname: name of the company to check for
       returns: number of files returned that pertain to companyname
       """
       i=0
       fob = open(filename,'r')
       lines = fob.readlines()
       for name in lines:
         if name.find(companyname)!=-1:
            i=i+1
       return i

def write_comparison_results(doc_base_sim,companyname, filename, reference_dir,method): 
       """
       Function: write_comparison_results
       --------------------------------------------------------------------------------
       doc_base_sim: (document-base-name, similarity) tuple list
       companyname: name of the company to check for
       filename: the output file to write the precision, recall values
       method: type of method used for similarity
       """     
       doc_list = list()
       i=0
       for x in doc_base_sim:
         t=x.split(',')
         doc_list.append(t[0])
         i=i+1
         if(i==100): break
       L1=i
       l=get_all_files(reference_dir)
       L2=len(l)
       i=0
       j=0
       for x in l:
         if x.find(companyname)!=-1:
           i=i+1
       relevant_cnt = i
       for y in doc_list:
         if y.find(companyname)!=-1:
           j=j+1
       retrieved_relevant = j
       precision = float(retrieved_relevant)/float(L1)
       recall =   float(retrieved_relevant)/float(relevant_cnt)
       f=open(filename,'a')
       if method=='tfidf':
         f.write('4.3b Starbucks_clustered_tfidf' + "\t" + str(precision) + "\t" + str(recall) + "\n")
       elif method=='mi':
         f.write('4.3b Starbucks_MI' + "\t" + str(precision) + "\t" + str(recall) + "\n")        
       f.close()
       

if __name__ == "__main__":
       l=load_collection_tokens('/home1/c/cis530/hw1/data/corpus/starbucks')
       tf=get_tf(l)  #dictionary
       tdl = gen2dlist('/home1/c/cis530/hw1/data/all_data')
       idf = get_idf(tdl)  #dictionary
       tfidf = get_tfidf_top(tf,idf,50)
       topic_terms = load_collection_tokens('/home1/c/cis530/hw1/data/corpus/starbucks')
       bg_terms = load_collection_tokens('/home1/c/cis530/hw1/data/corpus')
       mi = get_mi_top(bg_terms, topic_terms, 100)
       write_mi_weights(os.getcwd(),'starbucks_mi_weights.txt')
       val=tf.values()
       keys=tf.keys()
       vals_tf,keys_tf = quick_sort_recursive(val,keys, 0, len(val)-1)
       normalized_tf = list()
       i=0
       for x in keys_tf:
          normalized_tf.append(x) 
          i=i+1
          if(i==100): break
       a1=get_precision(normalized_tf,tfidf)
       a2=get_recall(normalized_tf, tfidf)
       b1=get_precision(mi, tfidf)
       b2=get_recall(mi, tfidf)
       f=open(os.getcwd() + "/" + "results.txt",'a')
       f.write("2.3 MI "  + str(b1) + "\t" + str(b2) + "\n")
       f.write("2.3 TFIDF " + str(a1) + "\t" + str(a2) + "\n")
       f.close()
       bc_dict = read_brown_cluster()
       brown_idfcl = get_idf_clusters(bc_dict)
       write_tfidf_weights(os.getcwd(),'starbucks_tfidf_weights.txt')
       mi_list = rank_doc_sim(os.getcwd() + "/" + "starbucks_mi_weights.txt",'mi','/home1/c/cis530/hw1/data/mixed', bc_dict)
       write_comparison_results(mi_list,'starbucks', 'results.txt', '/home1/c/cis530/hw1/data/mixed/','mi')
       tfidf_list = rank_doc_sim(os.getcwd() + "/" + "starbucks_tfidf_weights.txt", 'tfidf','/home1/c/cis530/hw1/data/mixed/', bc_dict)
       write_comparison_results(tfidf_list,'starbucks', 'results.txt', '/home1/c/cis530/hw1/data/mixed/','tfidf')
 
      
