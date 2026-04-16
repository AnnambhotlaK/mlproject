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

// Function to interpret prediction as who will win the match (player 1 or 2)
function interpretPrediction() {
    const predictionElement = document.getElementById('prediction-value');
    if (predictionElement) {
        const predictionValue = parseFloat(predictionElement.textContent);
        const percentageValue = (predictionValue * 100).toFixed(2);
        if (percentageValue > 50) {
            document.getElementById('interpretation').textContent = 'Player 2 is favored to win by ' + (percentageValue - 50) + '%';
        }
        else {
            document.getElementById('interpretation').textContent = 'Player 1 is favored to win by ' + (50 - percentageValue) + '%';
        }
    }
}

// Run the function when the DOM is loaded
document.addEventListener('DOMContentLoaded', formatPredictionAsPercentage);
document.addEventListener('DOMContentLoaded', interpretPrediction);