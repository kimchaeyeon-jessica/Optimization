stock_history=[]
max_sell=0
max_lead=2
average_lead=1
safe_stock=0
safe_period=0

while True:
    buy=int(input("Input buy amount"))
    sell=int(input("Input sell amount"))
    remain=0
    past_stocks=stock_history[-3:]
    item_count=len(past_stocks)+1
    sell_average=0
    last_stock=0
    total_sell=sell
    for past in past_stocks:
        total_sell=total_sell+past['sell']
        last_stock=past['stock']
    sell_average=total_sell//item_count

    if max_sell<sell:
        max_sell=sell
    if len(stock_history)==0:
        remain=buy-sell
        last_stock=remain
    else:
        remain=buy-sell+last_stock

    stock_period=remain/sell_average
    safe_stock=round(max_sell*max_lead-sell_average*average_lead)
    safe_period=round(safe_stock/sell_average,1)
    buy_recommend = round(safe_period*sell_average-last_stock+sell)
    if buy_recommend < 0:
        buy_recommend = 0
    stock={
        'buy':buy,
        'sell':sell,
        'stock':remain,
        'stock_period':stock_period,
        }
    stock_history.append(stock)
    print("구매량: %d" %(buy,))
    print("판매: %d" %(sell,))
    print("평균 판매: %d" %(sell_average,))
    print("재고: %d" %(remain,))
    print("재고주기: %f" %(stock_period,))
    print("안전 재고: %d" %(safe_stock,))
    print("안전 재고 주기: %f" %(safe_period,))
    print("추천 구매량: %d" %(buy_recommend,))
    
    
