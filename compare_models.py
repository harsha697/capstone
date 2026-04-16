import matplotlib.pyplot as plt

models = ["Random Forest", "XGBoost", "Ensemble"]
accuracy = [0.91, 0.93, 0.95]  # replace with your actual results

plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.show()