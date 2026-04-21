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
    const interpretationElement = document.getElementById('interpretation');
    
    if (predictionElement && interpretationElement) {
        const predictionPercentage = parseFloat(predictionElement.textContent);
        
        if (!isNaN(predictionPercentage)) {
            console.log('Prediction percentage:', predictionPercentage);
            
            if (predictionPercentage > 50) {
                interpretationElement.textContent = 'Player 2 is favored to win by ' + (predictionPercentage - 50).toFixed(2)
            } else {
                interpretationElement.textContent = 'Player 1 is favored to win by ' + (50 - predictionPercentage).toFixed(2)
            }
            interpretationElement.textContent += '%.';
        }
    }
}

// Run the function when the DOM is loaded
document.addEventListener('DOMContentLoaded', formatPredictionAsPercentage);
document.addEventListener('DOMContentLoaded', interpretPrediction);