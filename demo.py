
import turicreate as turi
import os

model_name = "<Name of your coreml model>"
data_url = "<folder name of your training data>"
label_name = "<name of your label>"
model = "resnet-50"
sframe = "<sframe>"

data = turi.image_analysis.load_images(data_url)

dir_path = len(os.path.dirname(os.path.realpath(__file__)) + data_url) + 1

def get_label(path):
        return path[dir_path]

data[label_name] = data["path"].apply(get_label)
data.save(sframe + ".sframe")
data.explore()

dataBuffer = turi.SFrame(sframe + ".sframe")

trainingBuffers, testingBuffers = dataBuffer.random_split(0.9)

model = turi.image_classifier.create(trainingBuffers, target=label_name, model=model)

evaluations = model.evaluate(testingBuffers)

print(evaluations["accuracy"])

model.save(model_name + ".model")

model.export_coreml(model_name + ".mlmodel")
# ###
