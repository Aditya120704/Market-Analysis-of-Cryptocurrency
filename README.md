# Crypto Analysis Application

## Project Overview

This application provides real-time and historical analysis of the cryptocurrency market. It offers a comprehensive suite of tools for investors, traders, and enthusiasts to monitor market trends, analyze sentiment, and make informed decisions.

## Features

*   **Market Overview:** A real-time dashboard that displays live market data, trending coins, and global market statistics.
*   **Price Prediction:** A sophisticated prediction model that forecasts future cryptocurrency prices using advanced machine learning techniques.
*   **Sentiment Analysis:** Real-time sentiment analysis of crypto news headlines, as well as overall market and individual currency sentiment.
*   **Advanced Analysis:** A comprehensive analysis of the relationship between the cryptocurrency market and the broader financial markets, including correlation with the S&P 500.
*   **Individual Currency Analysis:** Detailed analysis of individual cryptocurrencies, including historical data, volatility, and technical indicators.

## Architecture

The application is built with a modern, decoupled architecture that consists of a React frontend and a Flask backend.

*   **Frontend:** The frontend is a single-page application built with React, TypeScript, and Tailwind CSS. It uses Chart.js for data visualization and Axios for making API requests.
*   **Backend:** The backend is a Python-based API built with Flask. It provides a set of RESTful endpoints for accessing market data, sentiment analysis, and price predictions.
*   **Data:** The application uses a variety of data sources, including CoinGecko for market data, NewsAPI for news headlines, and a custom-trained sentiment analysis model.

## Getting Started

### Prerequisites

*   Node.js and npm
*   Python 3.x and pip

### Installation

1.  **Frontend:**
    ```bash
    cd frontend
    npm install
    ```
2.  **Backend:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Backend Server:**
    ```bash
    cd backend
    python app.py
    ```
2.  **Start the Frontend Development Server:**
    ```bash
    cd frontend
    npm start
    ```

## Testing

This project uses a comprehensive testing strategy that includes:

*   **Unit Tests:** To verify the functionality of individual components and functions.
*   **Integration Tests:** To ensure that the frontend and backend are working together correctly.
*   **End-to-End Tests:** To simulate real-world user scenarios and ensure the application is working as expected.

To run the tests, use the following command:

```bash
cd frontend
npm test
```
