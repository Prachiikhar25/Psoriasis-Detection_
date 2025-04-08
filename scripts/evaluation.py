import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/psoriasis_severity_model.h5')

# Predict the classes for the validation set
validation_generator.reset()
predictions = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Print classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Print confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)