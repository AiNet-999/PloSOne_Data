import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


stock_symbols = [
    'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUV', 
    'M', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 
    'MET', 'MGM', 'MHK', 'MKC', 'MKTX'
]



stocks=stock_symbols=companies = ["ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", 
                                  "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV",
                                  "SWK", "SBUX", "STT", "STLD"]

stock_symbols= [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB",
    "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", 
    "AMCR", "AEE"
]



plt.style.use('seaborn-muted')  
import seaborn as sns
sns.set_style("whitegrid", {
    "grid.color": "black",
    "grid.linestyle": ":",
    "axes.facecolor": "white"
})

x_test, y = next(test_dataset.as_numpy_iterator()) 
y_pred = model.predict(x_test)  # Get the model predictions


y_actual = scaler.inverse_transform(y[:, 0, :])  
y_pred = scaler.inverse_transform(y_pred[:, 0, :])  


fig, axes = plt.subplots(3, 5, figsize=(40, 20))

axes = axes.flatten()
n=45

for m in range(15):
    ax = axes[m]  # Get the current subplot axis
    ax.plot(y_actual[:, n], label="Actual", color='darkblue', linestyle='solid', linewidth=3)  
    ax.plot(y_pred[:, n], label="Forecast", color='darkorange', linestyle='--', linewidth=3)  
    ax.set_title(stock_symbols[m], fontsize=30) 
    ax.legend(loc='best', fontsize=20)  
    ax.grid(True, linestyle=':', color='gray', alpha=0.9)  
    ax.set_xlabel('Timesteps', fontsize=25) 
    ax.set_ylabel('Price', fontsize=25)  
    
 
    ax.tick_params(axis='x', labelsize=18, rotation=45)  
    ax.tick_params(axis='y', labelsize=18)  


    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    n=n+1


plt.tight_layout(pad=2.0)
plt.savefig("2.png")
plt.show()
