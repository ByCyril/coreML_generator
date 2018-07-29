
import coremltools.models as core
import turicreate as tc

model_name = input('Model Name: ')
file_name = input('Name of training data: ')
author = input('Author: ')
license = input('License: ')
short_description = input('Short Description: ')

data = tc.SFrame(file_name) # formatting our data to an SFrame

model = tc.sentence_classifier.create(data, 'Sentiment', features=['SentimentText'])

model.save(model_name + '.model')

model.export_coreml(model_name + '.mlmodel')

mlmodel = core.MLModel(model_name + '.mlmodel')

mlmodel.author = author
mlmodel.license = license
mlmodel.short_description = short_description

mlmodel.save(model_name + '.mlmodel')

