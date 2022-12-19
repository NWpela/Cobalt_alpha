# Scalar EMA

## Info

This alpha aims to predict future prices using previous patterns from the historical data.
To do so, the current window of Spot - EMA (called baseline) is compared to the previous ones using scalar product.
The forecast is the sum of the EMA prediction and the baseline prediction.

## Dependencies

- pandas
- ta
- 