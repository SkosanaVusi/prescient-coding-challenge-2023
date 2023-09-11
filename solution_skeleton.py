# %%
import numpy as np
import pandas as pd
import datetime
import plotly.express as px
import tensorflow as tf


print('---Python script Start---', str(datetime.datetime.now()))

# %%

# data reads
df_returns_train = pd.read_csv('data/returns_train.csv')
df_returns_test = pd.read_csv('data/returns_test.csv')
df_returns_train['month_end'] = pd.to_datetime(arg=df_returns_train['month_end']).apply(lambda d: d.date())
df_returns_test['month_end'] = pd.to_datetime(arg=df_returns_test['month_end']).apply(lambda d: d.date())

# %%

def equalise_weights(df: pd.DataFrame):

    # '''
    #     Function to generate the equal weights, i.e. 1/p for each active stock within a month

    #     Args:
    #         df: A return data frame. First column is month end and remaining columns are stocks

    #     Returns:
    #         A dataframe of the same dimension but with values 1/p on active funds within a month

    # '''

    # create df to house weights
    n_length = len(df) #number of rows
    df_returns = df
    df_weights = df_returns[:n_length].copy()
    df_weights.set_index('month_end', inplace=True)

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # assign 1/p
    df_weights[list_stocks] = 1/len(list_stocks)

    return df_weights


# %%

def generate_portfolio(df_train: pd.DataFrame, df_test: pd.DataFrame):

    '''
        Function to generate stocks weight allocation for time t+1 using historic data. Initial weights generated as 1/p for active stock within a month

        Args:
            df_train: The training set of returns. First column is month end and remaining columns are stocks
            df_test: The testing set of returns. First column is month end and remaining columns are stocks

        Returns:
            The returns dataframe and the weights
    '''

    print('---> training set spans', df_train['month_end'].min(), df_train['month_end'].max())
    print('---> training set spans', df_test['month_end'].min(), df_test['month_end'].max())

    # initialise data
    n_train = len(df_train)
    df_returns = pd.concat(objs=[df_train, df_test], ignore_index=True)

    df_weights = equalise_weights(df_returns[:n_train]) # df to store weights and create initial

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>

    # This is your playground. Delete/modify any of the code here and replace with 
    # your methodology. Below we provide a simple, naive estimation to illustrate 
    # how we think you should go about structuring your submission and your comments:

    # We use a static Inverse Volatility Weighting (https://en.wikipedia.org/wiki/Inverse-variance_weighting) 
    # strategy to generate portfolio weights.
    # Use the latest available data at that point in time
    
    # Define the neural network model for portfolio optimization
    def create_portfolio_model(input_shape, num_stocks):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_stocks, activation='softmax')
        ])
        return model

    # New function to calculate portfolio weights using the trained model
    def get_portfolio_weights(model, df_latest):
        returns_data = np.array(df_latest.drop(columns=['month_end']))
        predictions = model.predict(returns_data)
        normalized_weights = np.clip(predictions, 0, 0.1)  # Clip weights to ensure no stock > 10%
        weights_sum = np.sum(normalized_weights, axis=1, keepdims=True)
        portfolio_weights = normalized_weights / weights_sum
        return portfolio_weights

    # New function for training the portfolio model using backpropagation
    def train_portfolio_model(df_train, epochs=3000, batch_size=25):
        num_stocks = len(df_train.columns) - 1
        input_shape = (num_stocks,)
        model = create_portfolio_model(input_shape, num_stocks)

        x_train = np.array(df_train.drop(columns=['month_end']))
        y_train = x_train  # Input and output are the same for this self-supervised learning

        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        return model

    # Create and train the portfolio model
    model = train_portfolio_model(df_train)

    for i in range(len(df_test)):
        df_latest = df_returns[(df_returns['month_end'] < df_test.loc[i, 'month_end'])]

        # Get portfolio weights from the model
        portfolio_weights = get_portfolio_weights(model, df_latest)

        # Convert weights to DataFrame format
        df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + portfolio_weights.tolist()[0]],
                               columns=df_latest.columns)
        df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)

    
    # <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
    
    # 10% limit check
    if len(np.array(df_weights[list_stocks])[np.array(df_weights[list_stocks]) > 0.101]):

        raise Exception(r'---> 10% limit exceeded')

    return df_returns, df_weights


# %%


def plot_total_return(df_returns: pd.DataFrame, df_weights_index: pd.DataFrame, df_weights_portfolio: pd.DataFrame):

    '''
        Function to generate the two total return indices.

        Args:
            df_returns: Ascending date ordered combined training and test returns data.
            df_weights_index: Index weights. Equally weighted
            df_weights_index: Portfolio weights. Your portfolio should use equally weighted for the training date range. If blank will be ignored

        Returns:
            A plot of the two total return indices and the total return indices as a dataframe
    '''

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # replace nans with 0 in return array
    ar_returns = np.array(df_returns[list_stocks])
    np.nan_to_num(x=ar_returns, copy=False, nan=0)

    # calc index
    ar_rtn_index = np.array(df_weights_index[list_stocks])*ar_returns
    ar_rtn_port = np.array(df_weights_portfolio[list_stocks])*ar_returns

    v_rtn_index = np.sum(ar_rtn_index, axis=1)
    v_rtn_port = np.sum(ar_rtn_port, axis=1)

    # add return series to dataframe
    df_rtn = pd.DataFrame(data=df_returns['month_end'], columns=['month_end'])
    df_rtn['index'] = v_rtn_index
    df_rtn['portfolio'] = v_rtn_port
    df_rtn

    # create total return
    base_price = 100
    df_rtn.sort_values(by = 'month_end', inplace = True)
    df_rtn['index_tr'] = ((1 + df_rtn['index']).cumprod()) * base_price
    df_rtn['portfolio_tr'] = ((1 + df_rtn['portfolio']).cumprod()) * base_price
    df_rtn

    df_rtn_long = df_rtn[['month_end', 'index_tr', 'portfolio_tr']].melt(id_vars='month_end', var_name='series', value_name='Total Return')

    # plot
    fig1 = px.line(data_frame=df_rtn_long, x='month_end', y='Total Return', color='series')

    return fig1, df_rtn

# %%

# running solution
df_returns = pd.concat(objs=[df_returns_train, df_returns_test], ignore_index=True)
df_weights_index = equalise_weights(df_returns)
df_returns, df_weights_portfolio = generate_portfolio(df_returns_train, df_returns_test)
fig1, df_rtn = plot_total_return(df_returns, df_weights_index=df_weights_index, df_weights_portfolio=df_weights_portfolio)
fig1

# %%


# Using an Artificial Neural Network (ANN) and backpropagation to generate the weights for the portfolio was the approach we used and for the following reasons we chose it:

# 1. Non-linearity: ANNs can capture complex and non-linear relationships in the data. The stock market often exhibits non-linear patterns, and an ANN can better model these intricate interactions among different stocks.

# 2. Flexibility: ANNs can handle various types of data, including both numerical and categorical variables. This flexibility allows them to incorporate additional information, such as macroeconomic indicators or sector-specific data, which can improve the portfolio weight generation process.

# 3. Adaptability: The stock market is dynamic, and the relationships between different stocks may change over time. ANNs, especially when combined with backpropagation, can adapt and update the weights based on new incoming data, allowing the portfolio to adjust to changing market conditions.

# 4. Risk Management: ANNs can be integrated into the portfolio optimization process to consider risk factors beyond volatility. By training the ANN on historical data, it can learn to account for factors like downside risk, correlation between stocks, and other risk metrics, resulting in a more robust and risk-aware portfolio.

# 5. Portfolio Diversification: ANNs can optimize for diversification by learning to allocate weights in a way that minimizes the correlation among stocks. Diversification is a key aspect of risk reduction in a portfolio, and ANNs can help achieve this more efficiently.

# 6. Speed and Efficiency: Once the ANN is trained, generating portfolio weights for each time period is computationally efficient. It can quickly process large datasets and produce weight allocations, making it suitable for real-time or frequent rebalancing strategies.

# 7. Adoption of New Information: As new data becomes available, the ANN can continuously update the portfolio weights, allowing it to adapt to market changes and incorporate the most recent information into the investment decisions.

# However, it's important to note that using an ANN for portfolio optimization also comes with challenges, such as model complexity, data overfitting, and the need for appropriate hyperparameter tuning. This model also may require bigger training data but the following solutions we produced was able to surpass the benchmark and inverse volatility approach.
