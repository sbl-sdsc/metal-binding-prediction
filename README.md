# Protein Metal Binding Site Prediction
Methods to predict metal binding sites in a protein by it's amino acid sequence

## The Repository
[datasets](https://github.com/sbl-sdsc/metal-binding-prediction/tree/master/datasets) contains all required parquet data files. 
* **Metal_all_20180116.snappy.parquet** - is the raw dataframe
* **Metal_all_20180601.parquet** - is the processed dataframe 
* **Metal_all_20180601_predicted.parquet** - is the processed dataframe with predicted ligandId for all sequences

[dictionaries](https://github.com/sbl-sdsc/metal-binding-prediction/tree/master/dictionaries) contains all dictionaries for sequence encoding.

[logs](https://github.com/sbl-sdsc/metal-binding-prediction/tree/master/logs) contains trained F1 score records as charts (*.png) and data (results.txt)
* **results.txt** - format: phrase,LigandID,optimizer,learningrate,lossfunction,threshold for MBS prediction,#epochs,F1

[models](https://github.com/sbl-sdsc/metal-binding-prediction/tree/master/models) contains all saved models (*.json) and weights (*.h5) for all ligandIds. These are used with Keras.

[root folder](https://github.com/sbl-sdsc/metal-binding-prediction)
* [modules.py](https://github.com/sbl-sdsc/metal-binding-prediction/blob/master/modules.py) stores all functions (encoded data generators, trainer, etc)

* [metal_prediction.ipynb](https://github.com/sbl-sdsc/metal-binding-prediction/blob/master/metal_prediction.ipynb) train a keras model that predicts what type of metal ion a sequence binds to (first step)

* [MBS_prediction.ipynb](https://github.com/sbl-sdsc/metal-binding-prediction/blob/master/MBS_prediction.ipynb) train a keras model that predicts where in a sequence a metal ion binds to (second step)
	
* [predictor.ipynb](https://github.com/sbl-sdsc/metal-binding-prediction/blob/master/predictor.ipynb) does the prediction for both steps (metal type and fingerprints)





