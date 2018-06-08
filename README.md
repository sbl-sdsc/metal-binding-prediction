# Protein Metal Binding Site Prediction
Methods to predict metal binding sites in a protein by it's amino acid sequence

## The Repository
**datasets** contains all required parquet data files. 
* **Metal_all_20180116.snappy.parquet** - is the raw dataframe
* **Metal_all_20180601.parquet** - is the processed dataframe 
* **Metal_all_20180601_predicted.parquet** - is the processed dataframe with predicted ligandId for all sequences

**dictionaries** contains all dictionaries for sequence encoding.

**logs** contains trained F1 score records as charts (*.png) and data (results.txt)
* **results.txt** - format: phrase,LigandID,optimizer,learningrate,lossfunction,threshold for MBS prediction,#epochs,F1

**models** contains all saved models (*.json) and weights (*.h5) for all ligandIds. These are used with Keras.

**modules.py** stores all functions (encoded data generators, trainer, etc)

**metal_prediction.ipynb** predicts what type of metal ion a sequence binds to (first step)

**MBS_prediction.ipynb** predicts where in a sequence a metal ion binds to (second step)
	


	

predictor.ipynb





