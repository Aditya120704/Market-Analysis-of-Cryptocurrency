# Project Improvement Plan

This document outlines the future enhancements and features for the crypto analysis application.

## I. Prediction Model Enhancements

The goal is to improve the precision and accuracy of the price prediction model.

### 1. Advanced Feature Engineering
- [x] **Technical Indicators:**
  - [x] Moving Averages (SMA, EMA)
  - [x] Relative Strength Index (RSI)
  - [x] Moving Average Convergence Divergence (MACD)
  - [x] Bollinger Bands
- [x] **Sentiment Data Integration:**
  - [x] Feed the Bitcoin sentiment score into the prediction model.
  - [x] Feed the future overall market sentiment score into the model.
- [x] **On-Chain Metrics:**
  - [x] Active Wallet Addresses (Skipped for now)
  - [x] Transaction Volume & Count (Skipped for now)
  - [x] Network Hash Rate (Skipped for now)
- [x] **Macroeconomic Data:** (Deferred to Market Analysis phase)
  - [ ] S&P 500 / Nasdaq Composite
  - [ ] US Dollar Index (DXY)
  - [ ] Inflation Rates (CPI)

### 2. Advanced Model Exploration
- [x] **LSTM (Long Short-Term Memory Networks):** Implemented and tuned. Further tuning/architecture exploration needed.
- [ ] **Prophet:** Investigate for time-series with seasonal patterns.

## II. New Value-Added Features

Expand the application beyond basic data display and prediction.

- [ ] **Portfolio Tracker:**
  - Allow users to input their crypto holdings.
  - Track real-time portfolio value.
  - Display Profit & Loss (P&L) metrics.
  - Visualize asset allocation with a pie chart.
- [ ] **Real-Time Alerting System:**
  - Price alerts for specific crypto targets.
  - Sentiment shift alerts.
  - Technical indicator alerts (e.g., RSI crossing a threshold).
- [ ] **Market Correlation Matrix:**
  - A heatmap showing price correlations between different cryptocurrencies.
- [x] **Expanded Data Visualization:**
  - [ ] Overlay technical indicators on price charts.
  - [ ] Create dedicated charts for on-chain metrics.
  - [ ] Display results of various analyses (e.g., macroeconomic, correlation) on the Market Analysis page.

## III. Performance & Architecture

Address technical debt and improve the system's architecture.

- [x] **Sentiment Analysis Implementation:**
  - [x] Implemented real-time crypto news sentiment analysis using NewsAPI.
  - [x] Implemented overall market sentiment and individual currency sentiment.

## IV. UI/UX Enhancements (Ongoing)
- [ ] Continue refining the user interface for a professional, seamless look (e.g., Coinbase/Binance style).

## V. Sub-final Improvements
- [ ] **API Performance:**
  - [ ] Implement a caching mechanism for the sentiment analysis endpoint to avoid real-time recalculation on every call. Pre-calculate and store results in a JSON file.
- [ ] **Future Sentiment Features:**
  - [ ] Display top 5 currency overall sentiment on a daily basis (frontend aggregation/display).