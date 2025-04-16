def evaluate_model(model, test_data, metrics):
    predictions = model.predict(test_data['features'])
    results = {}

    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = accuracy_score(test_data['labels'], predictions)
        elif metric == 'f1_score':
            results['f1_score'] = f1_score(test_data['labels'], predictions, average='weighted')
        elif metric == 'confusion_matrix':
            results['confusion_matrix'] = confusion_matrix(test_data['labels'], predictions)

    return results

def visualize_results(results):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if 'confusion_matrix' in results:
        plt.figure(figsize=(10, 7))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    print("Evaluation Metrics:")
    for key, value in results.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")