import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def plot_boxplots(data, title):
    """Display boxplots for all columns in the DataFrame.

    Parameters:
        - data: DataFrame
    """
    plt.figure(figsize=(8, 5))
    data.boxplot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title(title)
    plt.ylabel("Values")
    plt.show()


def plot_pie_chart(data, column, colors=None):
    """Display a pie chart for a specific column in the DataFrame.

    Parameters:
        - data: DataFrame
        - column: str, the column to plot
        - colors: list, default=None
    """
    data = data[column].value_counts()
    labels = data.index
    sizes = data.values
    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f"Distribution of {column} column", fontsize=16, fontweight='bold')
    plt.show()


def plot_correlation_heatmap(data, title):
    """Display a correlation heatmap.

    Parameters:
        - data: DataFrame
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()


def plot_barplot(data, col, title):
    """
    Plot a custom barplot using the provided DataFrame.

    Parameters:
        - data: DataFrame, the data to plot
        - col: str, the column to plot on the x-axis
    """
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=data, x=col, hue=col, palette="viridis", legend=False, edgecolor='black')
    plt.title(title)
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    max_count = data[col].value_counts().max()
    plt.ylim(0, max_count + max_count * 0.1)
    
    plt.show()

def plot_importances(data, title):
    """
    Plot a barplot of the data.
    
    Parameters:
        - data: DataFrame with 'mean' and 'std' columns
        - title: Title of the plot
    """
    plt.figure(figsize=(6, 4))

    if isinstance(data, pd.Series):
        ax = sns.barplot(x=data.index, y=data)
    else :
        ax = sns.barplot(x=data.index, y=data['mean'])
        x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        plt.errorbar(x=x_coords, y=y_coords, yerr=data["std"], fmt="none", c= "k")
    
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title(title)
    plt.xlabel("Variables")
    plt.ylabel("Values")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title):
    """Display a confusion matrix.

    Parameters:
        - y_true: list
        - y_pred: list
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title(title)
    plt.show()


def plot_roc_curve(fpr, tpr, auc):
    """Display a ROC curve.

    Parameters:
        - fpr: list
        - tpr: list
    """
    plt.figure(figsize=(6, 4))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, color='orange', label=f'ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.show()


def plot_pca(train_x, test_x, train_y, test_y):
    """Display a PCA plot.

    Parameters:
        - train_x: DataFrame, the train data
        - test_x: DataFrame, the test data
        - train_y: DataFrame, the train target
        - test_y: DataFrame, the test target
    """
    X = pd.concat([train_x, test_x], axis=0)
    y = pd.concat([train_y, test_y], axis=0)
    data_labels = ['Train'] * len(train_x) + ['Test'] * len(test_x)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 4))
    # Plot Train data
    plt.scatter(X_pca[[label == 'Train' for label in data_labels], 0], 
                X_pca[[label == 'Train' for label in data_labels], 1], 
                c=y[[label == 'Train' for label in data_labels]], 
                label="Train", 
                edgecolor='k', alpha=0.7)
    # Plot Test data
    plt.scatter(X_pca[[label == 'Test' for label in data_labels], 0], 
                X_pca[[label == 'Test' for label in data_labels], 1], 
                c=y[[label == 'Test' for label in data_labels]], 
                label="Test", 
                edgecolor='k', alpha=0.7)
    plt.title("PCA Projection of the data (Train and Test)")
    plt.legend()
    plt.show()