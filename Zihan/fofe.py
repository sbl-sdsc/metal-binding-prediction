import numpy as np

class FofeVectorizer():

    def __init__( self, alpha = 0.99 ):
        self.alpha = alpha

    def naive_transform( self, docs, vocabulary ):
        x = np.zeros(( len( docs ), len( vocabulary )))
        for row_i, doc in enumerate( docs ):
            for word in doc:
                x[row_i,:] *= self.alpha
                try:
                    col_i = vocabulary[word]
                except KeyError:
                    continue
                x[row_i, col_i] += 1
        return x
