import csv
import numpy as np
import matplotlib.pyplot as plt

sale_data = []
optimal_xaxis = []
optimal_yaxis = []

with open("sample_data.csv", encoding='utf-8-sig')as data:
    reader=csv.DictReader(data)
    prices = []
    quantities = []
    for row in reader:
        price = int(row['price'])
        sale_qty = int(row['sale_qty'])
        sale_data.append({
            'price': price,
            'qty' : sale_qty,
        })
        prices.append(price)
        quantities.append(sale_qty)
        plt.scatter(price, sale_qty)
        
    x = np.array(prices)
    y = np.array(quantities)

    fit = np.polyfit(x, y, 2)
    print(fit)
    #polyfit은 다차 방정식에 대한 최적값을 회귀분석해줌
    #np.array는 numpy가 사용하는 데이터 구조로 변형하는 것
    #polyfit(x축 데이터, y축 데이터, 차수)
    for price in range(10000,100000,1000):
        optimal_xaxis.append(price)
        optimal_yaxis.append(fit[0] * (price ** 2) + fit[1] * price + fit[2])
        #ax^2+bx+c의 형태
    plt.plot(optimal_xaxis, optimal_yaxis)
    plt.show()
