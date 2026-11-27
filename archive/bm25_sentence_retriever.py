from pyserini.search.lucene import LuceneSearcher

class SentenceBM25Retriever():
  def __init__(self,  k1=1.2, b=0.4, use_rm3 = True):
    self.s = LuceneSearcher(r"C:\Users\mukun\crisis-claim-analysis\artifacts\bm25_sentences\index")
    self.s.set_bm25(k1, b)
    if use_rm3:
      self.s.set_rm3()

  def search(self, query, k = 50):
    hits = self.s.search(query, k = k)
    out = []
    #{"id": "Obama#0", "contents": "Barack Obama was born..."}
    #{"id": "Obama#1", "contents": "He served as senator..."}
    #h.docid = Obama#0, Obama#1 ...
    for hit in hits:
      docid = hit.docid
      if "#" in docid:
        page, line = docid.rsplit("#", 1)
        try: 
          line = int(line)
        except:
          line = -1
      else:
        page, line = docid, -1
      
      out.append({"id": page, "line": line, "score": float(hit.score)})
    
    return out
