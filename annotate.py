from roboflow import Roboflow
rf = Roboflow(api_key="VhKbMWBpCXxQ0zrwOW1T")
project = rf.workspace().project("productsdetectionindia")
model = project.version(1).model

model.predict("template.png", confidence=40, overlap=30).save("prediction.png")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())