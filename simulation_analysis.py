with open("random_forest_result.txt", 'r') as analysis_file:
    y_pred = []
    y_test = []
    for l in analysis_file:
        line = l.strip("[] \n")
        line_list = line.split()
        y_pred.append(float(line_list[0]))
        y_test.append(float(line_list[1]))

print(y_pred)
print(y_test)

money = 1000000
days = 0
transactions = 0

for i, x in enumerate(y_pred):
    if x > 1:
        print(f"predicted buy signal: {x}%. You bought.")
        money *= (1 + (y_test[i]-0.5)/100)
        days += 3
        transactions += 1
        print(f"after waiting for 3 days and paying transaction fee, "
              f"you gained/lost {y_test[i]-0.5:.2f}%. You now have {money:.1f}")
        print("-" * 50)

end_profit_percent = (money-10**6)/(10**6)*100
end_multiplier = money/(10**6)
exponential_base_of_profit = end_multiplier**(1/transactions)
print(f"in the end, after {days} days and {transactions} transactions, you gained a profit "
      f"of {end_profit_percent:.2f}%.")
print(f"average profit per day = {end_profit_percent/days:.2f}%")
print(f"exponential average profit per transactoion = {exponential_base_of_profit:.4f}")


