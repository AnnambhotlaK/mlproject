// script.js - JavaScript functions for home.html

// Function to format prediction as percentage rounded to two decimal points
function formatPredictionAsPercentage() {
    const predictionElement = document.getElementById('prediction-value');
    if (predictionElement) {
        const predictionValue = parseFloat(predictionElement.textContent);
        if (!isNaN(predictionValue)) {
            predictionElement.textContent = (predictionValue * 100).toFixed(2);
        }
    }
}

// Run the function when the DOM is loaded
document.addEventListener('DOMContentLoaded', formatPredictionAsPercentage);