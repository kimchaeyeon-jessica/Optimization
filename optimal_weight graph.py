import csv
import matplotlib.pyplot as plt

sale_data=[] #파일을 읽고나서 데이터를 보관하는 장소
optimal_weight =0 #최적의 기울기
min_difference = -1 #차이 최솟값

with open("sample_data.csv", encoding='utf-8-sig')as data:
    reader=csv.DictReader(data)

    for row in reader:
        price = int(row['price'])
        sale_qty = int(row['sale_qty'])
        sale_data.append({
            'price': price,
            'qty' : sale_qty,
        }) #판매 데이터
        plt.scatter(price, sale_qty)

    for denominator in range(-100,101):
        if denominator == 0:
            continue
        for numerator in range(1,101):
            weight = numerator / (denominator * 1000) #1000은 그냥 보정값

            sum_difference = 0
            for sale in sale_data:
                estimate_qty = abs(weight * sale.get('price'))
                difference = abs(estimate_qty - sale.get('qty'))
                sum_difference += difference

            if min_difference < 0 or min_difference > sum_difference:
                min_difference = sum_difference
                optimal_weight = weight

    optimal_xaxis = []
    optimal_yaxis = []
    for price in range(10000,100000,1000):
        optimal_xaxis.append(price)
        optimal_yaxis.append(optimal_weight * price)
    print(optimal_weight)

    plt.plot(optimal_xaxis, optimal_yaxis)
    plt.show()
        
   
