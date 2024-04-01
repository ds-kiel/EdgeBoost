import numpy as np
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def plot_calibration_curve(y_test, mobnet_uncal, mobnet_cal):

    num_classes = 100 # Replace with the number of classes in your multiclass problem
    test_labels = label_binarize(y_test, classes=np.arange(num_classes))

    # Step 2: Calculate the reliability diagram for mobnet_cal
    prob_true, prob_pred = calibration_curve(test_labels.ravel(), mobnet_uncal.ravel(), n_bins=15)

    # Calculate the reliability diagram for probs2
    prob_true2, prob_pred2 = calibration_curve(test_labels.ravel(), mobnet_cal.ravel(), n_bins=15)

    # Create a figure and set the font size
    plt.figure()
    plt.rc('font', size=16)  # Set font size for all elements

    # Step 3: Calculate Expected Calibration Error (ECE) for both sets of probabilities
    ece = np.abs(prob_pred - prob_true).mean()
    ece2 = np.abs(prob_pred2 - prob_true2).mean()

    # Step 4: Plot the reliability diagrams for both sets of probabilities
    plt.plot(prob_pred, prob_true, marker='o', color='purple', linestyle='-', label='Poor Calibration')
    plt.plot(prob_pred2, prob_true2, marker='*', color='purple', linestyle='--', label='Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    plt.xlabel('Classifier Probabilities')
    plt.ylabel('Empirical Accuracy')
    #plt.title('MobileNetv3 Small Trained on ImageNet', fontweight='bold', fontsize=18)  # Adjust title font size
    #plt.legend(frameon=False,loc='upper left', fontsize=17, bbox_to_anchor=(0.00001, 1))  # Adjust legend font size
    plt.tight_layout()  # Increase padding

    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('cifar-reliability.pdf',bbox_inches='tight')

    plt.show()


def offload(y_test, mobnet_uncal, mobnet_cal, effnet):

    # Encode labels in one-hot format
    encoded_array = np.zeros((y_test.size, y_test.max()+1), dtype=int)
    encoded_array[np.arange(y_test.size), y_test] = 1

    # Thresholds for decision making
    T = [0.04, 0.087, 0.298, 0.59, 0.72, 0.83, 0.93]

    combined_accuracy = []
    perc_model_2_samples = []
    model1_accuracy = []

    for threshold in T:
        total_samples = correct_predictions = uncertain_predictions  = correct_model1 = 0
        
        for i in range(len(y_test)):
            y_true = encoded_array[i]
            y_pred_cal = mobnet_cal[i,:]
            diff = np.max(y_pred_cal) - np.sort(y_pred_cal)[-2]
            
            if diff >= threshold:
                total_samples += 1
                correct_predictions += y_true[np.argmax(y_pred_cal)] == 1
                correct_model1 += y_true[np.argmax(y_pred_cal)] == 1
            else:
                y_pred_effnet = effnet[i,:]
                total_samples += 1
                uncertain_predictions += 1
                correct_predictions += y_true[np.argmax(y_pred_effnet)] == 1

        # Compute metrics
        combined_acc = round((correct_predictions / total_samples) * 100, 1)
        combined_accuracy.append(combined_acc)
        perc_model_2 = round((uncertain_predictions / total_samples) * 100, 2)
        perc_model_2_samples.append(perc_model_2)
        model1_acc = (correct_model1 / (total_samples - uncertain_predictions)) * 100 if (total_samples - uncertain_predictions) > 0 else 0
        model1_accuracy.append(model1_acc)

        print(f'Combined approach accuracy on test set: {combined_acc}%')
        print(f'% of uncertain predictions at network 1: {perc_model_2}%')
        print(f'Total correct to model 1 out of passed: {model1_acc:.2f}%')
        print('---------------------------------------------------------------------------')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate combined model predictions for CIFAR100 dataset.")
    parser.add_argument("--mobnet_cal", type=str, required=True, help="Path to the calibrated MobileNet probabilities npy file.")
    parser.add_argument("--mobnet_uncal", type=str, required=True, help="Path to the uncalibrated MobileNet probabilities npy file.")
    parser.add_argument("--effnet", type=str, required=True, help="Path to the EfficientNet probabilities npy file.")
    parser.add_argument("--labels", type=str, required=True, help="Path to the labels npy file.")

    args = parser.parse_args()

       # Load the data
    effnet = np.load(args.effnet)
    mobnet_uncal = np.load(args.mobnet_uncal)
    mobnet_cal = np.load(args.mobnet_cal)
    y_test = np.load(args.labels)

    plot_calibration_curve(y_test, mobnet_uncal, mobnet_cal)
    offload(y_test, mobnet_uncal, mobnet_cal, effnet)


